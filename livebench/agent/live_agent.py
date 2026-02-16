"""
LiveAgent - Main agent class for LiveBench with decision-making framework
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import LiveBench components
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent.economic_tracker import EconomicTracker
from agent.message_formatter import format_tool_result_message, format_result_for_logging
from work.task_manager import TaskManager
from work.evaluator import WorkEvaluator
from prompts.live_agent_prompt import (
    get_live_agent_system_prompt,
    get_work_task_prompt,
    format_cost_update,
    STOP_SIGNAL
)
from livebench.utils.logger import LiveBenchLogger, set_global_logger

# Load environment variables
load_dotenv()


class LiveAgent:
    """
    LiveAgent - AI agent for economic survival simulation

    Core functionality:
    1. Economic tracking (balance, token costs, income)
    2. Daily decision-making (work vs learn)
    3. Work task execution
    4. Learning and knowledge accumulation
    5. Survival management
    """

    def __init__(
        self,
        signature: str,
        basemodel: str,
        initial_balance: float = 1000.0,
        input_token_price: float = 0.01,
        output_token_price: float = 0.03,
        max_work_payment: float = 50.0,
        mcp_config: Optional[Dict[str, Dict[str, Any]]] = None,
        data_path: Optional[str] = None,
        max_steps: int = 20,
        max_retries: int = 5,
        base_delay: float = 1.0,
        api_timeout: float = 60.0,
        openai_base_url: Optional[str] = None,
        # New task source parameters
        task_source_type: str = "parquet",
        task_source_path: Optional[str] = None,
        inline_tasks: Optional[List[Dict]] = None,
        # New filtering and assignment parameters
        agent_filters: Optional[Dict[str, List[str]]] = None,
        agent_assignment: Optional[Dict[str, Any]] = None,
        # Task value pricing
        task_values_path: Optional[str] = None,
        # Evaluation parameters
        use_llm_evaluation: bool = True,
        meta_prompts_dir: str = "./eval/meta_prompts",
        # Tasks per day parameter
        tasks_per_day: int = 1,
        # Multimodal support parameter
        supports_multimodal: bool = True
    ):
        """
        Initialize LiveAgent

        Args:
            signature: Agent signature/name
            basemodel: Base model name
            initial_balance: Starting balance in dollars
            input_token_price: Price per 1K input tokens
            output_token_price: Price per 1K output tokens
            max_work_payment: Maximum payment for work tasks (used as default if no task values)
            mcp_config: MCP tool configuration
            data_path: Path to store agent data
            max_steps: Maximum reasoning steps per session
            max_retries: Maximum retry attempts for API calls (default: 5)
            base_delay: Base delay in seconds for exponential backoff retries (default: 1.0)
            api_timeout: Timeout in seconds for each API call (default: 60.0)
            openai_base_url: OpenAI API base URL
            task_source_type: Type of task source ("parquet", "jsonl", or "inline")
            task_source_path: Path to task source file
            inline_tasks: List of inline tasks
            agent_filters: Filter criteria for task selection
            agent_assignment: Explicit task assignment configuration
            task_values_path: Path to task_values.jsonl with calculated task prices
            use_llm_evaluation: Whether to use LLM-based evaluation
            meta_prompts_dir: Path to evaluation meta-prompts directory
            tasks_per_day: Number of tasks agent can work on per day
            supports_multimodal: Whether the model supports multimodal (image) inputs
        """
        self.signature = signature
        # Allow overriding model via environment variable
        env_model = os.getenv("OPENAI_MODEL")
        if env_model:
            self.basemodel = env_model
            print(f"🔧 Using model from environment: {self.basemodel}")
        else:
            self.basemodel = basemodel
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.api_timeout = api_timeout
        self.tasks_per_day = tasks_per_day
        self.supports_multimodal = supports_multimodal

        # Set data path
        self.data_path = data_path or f"./livebench/data/agent_data/{signature}"
        
        # Initialize logger
        self.logger = LiveBenchLogger(signature=signature, data_path=self.data_path)
        set_global_logger(self.logger)

        # Set OpenAI configuration
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_API_BASE")

        # Initialize components
        self.economic_tracker = EconomicTracker(
            signature=signature,
            initial_balance=initial_balance,
            input_token_price=input_token_price,
            output_token_price=output_token_price,
            data_path=os.path.join(self.data_path, "economic")
        )

        # Initialize TaskManager with new parameters
        self.task_manager = TaskManager(
            task_source_type=task_source_type,
            task_source_path=task_source_path,
            inline_tasks=inline_tasks,
            task_data_path=self.data_path,
            agent_filters=agent_filters,
            agent_assignment=agent_assignment,
            task_values_path=task_values_path,
            default_max_payment=max_work_payment
        )

        self.evaluator = WorkEvaluator(
            max_payment=max_work_payment,
            data_path=self.data_path,
            use_llm_evaluation=use_llm_evaluation,
            meta_prompts_dir=meta_prompts_dir
        )

        # Set MCP configuration
        self.mcp_config = mcp_config or self._get_default_mcp_config()

        # MCP and AI components
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: Optional[List] = None
        self.model: Optional[ChatOpenAI] = None
        self.agent: Optional[Any] = None

        # Daily state
        self.current_date: Optional[str] = None
        self.current_task: Optional[Dict] = None
        self.daily_activity: Optional[str] = None  # "work" or "learn"
        self.daily_work_income: float = 0.0
        self.daily_trading_profit: float = 0.0

    def _get_default_mcp_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default MCP configuration - Work and Learn only"""
        config = {
            "livebench": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('LIVEBENCH_HTTP_PORT', '8010')}/mcp",
            }
        }
        # Trading functionality disabled - focusing on work and learn capabilities only
        return config

    async def initialize(self) -> None:
        """Initialize agent components"""
        print(f"🚀 Initializing LiveAgent: {self.signature}")

        # Initialize economic tracker
        self.economic_tracker.initialize()

        # Load tasks
        self.task_manager.load_tasks()

        # Get tools directly (no MCP)
        from livebench.tools.direct_tools import get_all_tools, set_global_state as set_tool_state

        self.tools = get_all_tools()
        print(f"✅ Loaded {len(self.tools)} LiveBench tools")

        # Set tool state
        set_tool_state(
            signature=self.signature,
            economic_tracker=self.economic_tracker,
            task_manager=self.task_manager,
            evaluator=self.evaluator,
            current_date=self.current_date,
            current_task=self.current_task,
            data_path=self.data_path,
            supports_multimodal=self.supports_multimodal
        )

        # Create AI model with custom httpx clients (bypass proxy)
        import httpx
        http_client_sync = httpx.Client(
            timeout=60.0,
            trust_env=False  # Don't use environment proxy settings
        )
        http_client_async = httpx.AsyncClient(
            timeout=60.0,
            trust_env=False
        )

        self.model = ChatOpenAI(
            model=self.basemodel,
            base_url=self.openai_base_url,
            max_retries=3,
            timeout=60,
            http_client=http_client_sync,
            http_async_client=http_client_async
        )

        print(f"✅ LiveAgent {self.signature} initialization completed")

    def _prepare_reference_files(self, date: str, task: Dict) -> List[str]:
        """
        Copy task reference files to agent's sandbox AND upload to E2B sandbox for code execution.
        
        Args:
            date: Current date
            task: Task dictionary with reference_files list (can be list or numpy array)
            
        Returns:
            List of remote paths in E2B sandbox (e.g., ["/home/user/reference_files/file.pdf"])
        """
        import shutil
        
        reference_files = task.get('reference_files', [])
        
        # Handle both list and numpy array (from pandas DataFrame)
        if reference_files is None:
            return []
        try:
            if len(reference_files) == 0:
                return []
        except (TypeError, AttributeError):
            # If len() fails, it's not a sequence
            return []
        
        # Get absolute paths to reference files
        ref_file_paths = self.task_manager.get_task_reference_files(task)
        
        # Create sandbox directory for reference files (host filesystem)
        sandbox_dir = os.path.join(self.data_path, "sandbox", date, "reference_files")
        os.makedirs(sandbox_dir, exist_ok=True)
        
        copied_files = []
        missing_files = []
        e2b_remote_paths = []
        
        for src_path in ref_file_paths:
            if os.path.exists(src_path):
                # Copy file to sandbox, preserving filename
                filename = os.path.basename(src_path)
                dest_path = os.path.join(sandbox_dir, filename)
                
                try:
                    shutil.copy2(src_path, dest_path)
                    copied_files.append(filename)
                    self.logger.debug(
                        f"Copied reference file: {filename}",
                        context={"src": src_path, "dest": dest_path},
                        print_console=False
                    )
                    
                    # Upload to E2B sandbox for execute_code access
                    try:
                        from livebench.tools.productivity.code_execution_sandbox import upload_task_reference_files
                        remote_paths = upload_task_reference_files([dest_path])
                        if remote_paths:
                            e2b_remote_paths.extend(remote_paths)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to upload {filename} to E2B sandbox: {str(e)}",
                            context={"file": filename},
                            print_console=False
                        )
                    
                except Exception as e:
                    self.logger.warning(
                        f"Failed to copy reference file: {filename}",
                        context={"src": src_path, "error": str(e)},
                        print_console=False
                    )
            else:
                missing_files.append(src_path)
                self.logger.warning(
                    f"Reference file not found: {src_path}",
                    context={"task_id": task.get('task_id')},
                    print_console=False
                )
        
        if copied_files:
            self.logger.terminal_print(f"📎 Copied {len(copied_files)} reference file(s) to sandbox")
            if e2b_remote_paths:
                self.logger.terminal_print(f"   📤 Uploaded {len(e2b_remote_paths)} file(s) to E2B sandbox")
            self.logger.info(
                "Reference files prepared",
                context={
                    "date": date,
                    "task_id": task.get('task_id'),
                    "copied": copied_files,
                    "missing": missing_files,
                    "e2b_paths": e2b_remote_paths
                },
                print_console=False
            )
        
        if missing_files:
            self.logger.terminal_print(f"⚠️ Warning: {len(missing_files)} reference file(s) not found")
        
        # Store E2B paths in task for prompt generation
        task['e2b_reference_paths'] = e2b_remote_paths
        return e2b_remote_paths

    def _setup_logging(self, date: str) -> str:
        """Set up log file path for activity messages"""
        log_path = os.path.join(self.data_path, 'activity_logs', date)
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, "log.jsonl")

    def _log_message(self, log_file: str, messages: List[Dict[str, str]]) -> None:
        """Log messages to log file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "signature": self.signature,
            "messages": messages
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    async def _ainvoke_with_retry(self, messages: List[Dict[str, str]], timeout: float = 120.0) -> Any:
        """
        Agent invocation with retry, timeout, and token tracking
        
        Args:
            messages: List of messages to send to the agent
            timeout: Maximum time in seconds to wait for API response (default: 120s)
            
        Returns:
            Agent response
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # Convert messages to LangChain format
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

                lc_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    # Handle multimodal content (list of content items) vs string content
                    # Multimodal messages have content as a list of dicts with type/text/image_url
                    # String messages have content as a simple string

                    if role == "system":
                        lc_messages.append(SystemMessage(content=content))
                    elif role == "assistant" or role == "ai":
                        lc_messages.append(AIMessage(content=content))
                    else:  # user or human
                        # LangChain HumanMessage can accept both string and list[dict] content
                        lc_messages.append(HumanMessage(content=content))

                # Invoke the model with explicit timeout
                try:
                    response = await asyncio.wait_for(
                        self.agent.ainvoke(lc_messages),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"API call timed out after {timeout} seconds")

                # Track token usage if available
                input_text = " ".join([m.get("content", "") for m in messages if isinstance(m.get("content"), str)])
                self._estimate_and_track_tokens(input_text, response)

                return response

            except Exception as e:
                error_type = type(e).__name__
                is_timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError))
                
                self.logger.warning(
                    f"Agent invocation attempt {attempt}/{self.max_retries} failed",
                    context={
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "error_type": error_type,
                        "is_timeout": is_timeout,
                        "message_count": len(messages)
                    },
                    print_console=True
                )
                
                if attempt == self.max_retries:
                    self.logger.error(
                        f"Agent invocation failed after {self.max_retries} attempts",
                        exception=e,
                        print_console=True
                    )
                    raise e
                    
                retry_delay = self.base_delay * attempt
                self.logger.terminal_print(f"⚠️ Attempt {attempt} failed ({error_type}), retrying in {retry_delay}s...")
                self.logger.terminal_print(f"   Error: {str(e)[:200]}")
                await asyncio.sleep(retry_delay)

    def _estimate_and_track_tokens(self, input_text: str, response: Any) -> None:
        """Estimate and track token usage"""
        # Simple estimation: ~4 characters per token
        input_tokens = len(input_text) // 4

        # Extract response text from output
        output_text = str(response.get("output", response)) if isinstance(response, dict) else str(response)
        output_tokens = len(output_text) // 4

        # Track tokens
        self.economic_tracker.track_tokens(input_tokens, output_tokens)

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments"""
        # Find the tool
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == tool_name:
                try:
                    # LangChain tools can be invoked directly
                    result = tool.invoke(tool_args)

                    # Print result to console and terminal log (format for logging to avoid binary data)
                    formatted_result = format_result_for_logging(result)
                    self.logger.terminal_print(f"   ✅ Result: {formatted_result}")
                    
                    # Log successful tool execution
                    self.logger.debug(
                        f"Tool executed successfully: {tool_name}",
                        context={"tool": tool_name, "args": str(tool_args)[:200]},
                        print_console=False
                    )

                    return result
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    self.logger.terminal_print(f"   ❌ {error_msg}")
                    
                    # Log tool execution error
                    self.logger.error(
                        f"Tool execution failed: {tool_name}",
                        context={"tool": tool_name, "args": tool_args},
                        exception=e,
                        print_console=False
                    )
                    
                    import traceback
                    traceback.print_exc()
                    return error_msg

        error = f"Tool {tool_name} not found"
        self.logger.terminal_print(f"   ❌ {error}")
        
        # Log tool not found error
        self.logger.error(
            f"Tool not found: {tool_name}",
            context={
                "tool": tool_name,
                "available_tools": [t.name for t in self.tools if hasattr(t, 'name')]
            },
            print_console=False
        )
        
        return error

    async def run_daily_session(self, date: str) -> Optional[str]:
        """
        Run daily session: decision-making and activity execution

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            "NO_TASKS_AVAILABLE" if no tasks left, "ERROR" on error, None on success
        """
        # Set up logging (both conversation and terminal logs)
        log_file = self._setup_logging(date)
        self.logger.setup_terminal_log(date)
        
        self.logger.terminal_print(f"\n{'='*60}")
        self.logger.terminal_print(f"📅 LiveBench Daily Session: {date}")
        self.logger.terminal_print(f"   Agent: {self.signature}")
        self.logger.terminal_print(f"{'='*60}\n")

        self.current_date = date
        self.daily_work_income = 0.0
        self.daily_trading_profit = 0.0

        # Check if bankrupt
        if self.economic_tracker.is_bankrupt():
            self.logger.terminal_print("💀 Agent is BANKRUPT! Cannot continue.")
            self.logger.error(
                "Agent is bankrupt and cannot continue",
                context={"date": date, "balance": self.economic_tracker.get_balance()},
                print_console=False
            )
            return

        # Select daily work task
        try:
            self.current_task = self.task_manager.select_daily_task(date, self.signature)
            if not self.current_task:
                self.logger.terminal_print("🛑 No tasks available - stopping agent")
                self.logger.info(
                    "Agent stopped: No more tasks available",
                    context={"date": date},
                    print_console=False
                )
                # Return special marker to indicate no tasks available
                return "NO_TASKS_AVAILABLE"
            else:
                # Start tracking costs for this task with the task's date
                self.economic_tracker.start_task(self.current_task['task_id'], date=date)
        except Exception as e:
            self.logger.error(
                f"Error selecting daily task for {date}",
                context={"date": date},
                exception=e,
                print_console=True
            )
            self.current_task = None
            return "ERROR"

        # Copy reference files to sandbox for agent access
        if self.current_task:
            ref_files = self.current_task.get('reference_files')
            # Handle both list and numpy array (from pandas)
            has_ref_files = False
            if ref_files is not None:
                try:
                    # Check if it has any elements (works for list, numpy array, etc.)
                    has_ref_files = len(ref_files) > 0
                except (TypeError, AttributeError):
                    # If len() fails, try truthiness (for non-sequence types)
                    has_ref_files = bool(ref_files)
            
            if has_ref_files:
                try:
                    self._prepare_reference_files(date, self.current_task)
                except Exception as e:
                    self.logger.error(
                        "Failed to prepare reference files",
                        context={"date": date, "task_id": self.current_task.get('task_id')},
                        exception=e,
                        print_console=True
                    )
                    # Don't fail the session, but agent won't have reference files

        # Update tool state with current date and task
        try:
            from livebench.tools.direct_tools import set_global_state as set_tool_state
            set_tool_state(
                signature=self.signature,
                economic_tracker=self.economic_tracker,
                task_manager=self.task_manager,
                evaluator=self.evaluator,
                current_date=date,
                current_task=self.current_task,
                data_path=self.data_path,
                supports_multimodal=self.supports_multimodal
            )
            
            # Log task assignment for debugging
            if self.current_task:
                self.logger.terminal_print(f"✅ Task state updated: {self.current_task.get('task_id', 'unknown')}")
                self.logger.info(
                    f"Task state set successfully",
                    context={
                        "date": date,
                        "task_id": self.current_task.get('task_id', 'unknown'),
                        "sector": self.current_task.get('sector', 'unknown')
                    },
                    print_console=False
                )
            else:
                self.logger.terminal_print(f"⚠️ WARNING: No task was selected for {date}")
                self.logger.warning(
                    f"Task state set with no task",
                    context={"date": date},
                    print_console=False
                )
        except Exception as e:
            self.logger.error(
                "Failed to set global tool state",
                context={"date": date},
                exception=e,
                print_console=True
            )
            raise

        # Create agent with today's system prompt
        economic_state = self.economic_tracker.get_summary()
        system_prompt = get_live_agent_system_prompt(
            date=date,
            signature=self.signature,
            economic_state=economic_state,
            work_task=self.current_task,
            max_steps=self.max_steps
        )

        # Bind tools to the model
        self.agent = self.model.bind_tools(self.tools)

        # Initial messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Today is {date}. Analyze your situation and decide your activity."}
        ]

        self._log_message(log_file, messages)

        # Agent reasoning loop with tool calling
        max_iterations = 15
        activity_completed = False

        for iteration in range(max_iterations):
            self.logger.terminal_print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")

            try:
                # Call agent with timeout and retry
                try:
                    response = await self._ainvoke_with_retry(messages, timeout=self.api_timeout)
                except Exception as api_error:
                    # API call failed after all retries - skip this task and continue
                    self.logger.terminal_print(f"\n❌ API call failed after {self.max_retries} retries")
                    self.logger.terminal_print(f"   Error: {str(api_error)[:200]}")
                    self.logger.terminal_print(f"   ⏭️ Skipping current task and continuing...")
                    self.logger.error(
                        f"API call failed, skipping task",
                        context={
                            "date": date,
                            "task_id": self.current_task.get('task_id') if self.current_task else None,
                            "iteration": iteration + 1,
                            "max_retries": self.max_retries
                        },
                        exception=api_error,
                        print_console=False
                    )
                    # End task tracking before breaking
                    try:
                        self.economic_tracker.end_task()
                    except Exception:
                        pass
                    # Break out of iteration loop to skip this task
                    break

                # Extract response content
                if hasattr(response, 'content'):
                    agent_response = response.content
                else:
                    agent_response = str(response)

                # Show agent thinking (truncated)
                if len(agent_response) > 200:
                    self.logger.terminal_print(f"💭 Agent: {agent_response[:200]}...")
                else:
                    self.logger.terminal_print(f"💭 Agent: {agent_response}")

                # Check for tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    self.logger.terminal_print(f"🔧 Tool calls: {len(response.tool_calls)}")

                    # Add AI message
                    messages.append({"role": "assistant", "content": agent_response})

                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})

                        self.logger.terminal_print(f"\n   📞 Calling: {tool_name}")
                        self.logger.terminal_print(f"   📥 Args: {str(tool_args)[:100]}...")

                        # Find and execute the tool
                        tool_result = await self._execute_tool(tool_name, tool_args)

                        # Check if activity was completed
                        if tool_name == 'submit_work':
                            # End task tracking
                            self.economic_tracker.end_task()
                            
                            # Check if work was successful and extract payment
                            result_dict = tool_result if isinstance(tool_result, dict) else {}
                            if 'actual_payment' in result_dict or 'payment' in result_dict:
                                try:
                                    if not isinstance(result_dict, dict):
                                        result_dict = eval(str(tool_result))
                                    # Use actual_payment which respects evaluation threshold
                                    actual_payment = result_dict.get('actual_payment', result_dict.get('payment', 0))
                                    evaluation_score = result_dict.get('evaluation_score', 0.0)
                                    
                                    if actual_payment > 0:
                                        self.daily_work_income += actual_payment
                                        self.logger.terminal_print(f"\n   💰 Earned: ${actual_payment:.2f} (Score: {evaluation_score:.2f})")
                                        activity_completed = True
                                    elif evaluation_score > 0:
                                        # Work was submitted but didn't meet quality threshold
                                        self.logger.terminal_print(f"\n   ⚠️  Quality score {evaluation_score:.2f} below threshold - no payment")
                                        activity_completed = True
                                except:
                                    pass
                            if 'success' in str(tool_result).lower():
                                activity_completed = True
                        elif tool_name == 'learn' and 'success' in str(tool_result).lower():
                            activity_completed = True

                        # Add tool result to messages (handle multimodal content)
                        tool_message = format_tool_result_message(
                            tool_name, tool_result, tool_args, activity_completed
                        )
                        messages.append(tool_message)
                    # If activity is completed, stop the loop
                    if activity_completed:
                        self.logger.terminal_print(f"\n✅ Activity completed successfully!")
                        break

                    # Continue loop to get next response
                    continue

                # No more tool calls - agent is done
                self._log_message(log_file, [{"role": "assistant", "content": agent_response}])
                self.logger.terminal_print(f"\n✅ Agent completed daily session")
                break

            except Exception as e:
                # Unexpected error (not from API call) - log and re-raise
                self.logger.terminal_print(f"\n❌ Unexpected error in daily session: {str(e)}")
                self.logger.error(
                    f"Unexpected error in daily session iteration {iteration + 1}",
                    context={
                        "date": date,
                        "iteration": iteration + 1,
                        "max_iterations": max_iterations,
                        "activity_completed": activity_completed
                    },
                    exception=e,
                    print_console=False
                )
                import traceback
                traceback.print_exc()
                raise

        # WRAP-UP WORKFLOW: If activity not completed, try to collect and submit artifacts
        if not activity_completed and self.current_task:
            self.logger.terminal_print("\n⚠️ Iteration limit reached without task completion")
            self.logger.terminal_print("🔄 Initiating wrap-up workflow to collect artifacts...")
            
            try:
                from livebench.agent.wrapup_workflow import create_wrapup_workflow
                
                # Create sandbox directory path for this date
                sandbox_dir = os.path.join(
                    self.data_path,
                    "sandbox",
                    date
                )
                
                # Create and run wrap-up workflow with conversation context
                wrapup = create_wrapup_workflow(llm=self.model, logger=self.logger)
                wrapup_result = await wrapup.run(
                    date=date,
                    task=self.current_task,
                    sandbox_dir=sandbox_dir,
                    conversation_history=messages  # Pass conversation for context
                )
                
                # Process results
                submission = wrapup_result.get("submission_result")
                if submission and isinstance(submission, dict):
                    if submission.get("success"):
                        payment = submission.get("payment", 0)
                        if payment > 0:
                            self.daily_work_income += payment
                            activity_completed = True
                            self.logger.terminal_print(f"\n✅ Wrap-up workflow succeeded! Earned: ${payment:.2f}")
                    else:
                        self.logger.terminal_print(f"\n⚠️ Wrap-up workflow completed but submission failed")
                else:
                    self.logger.terminal_print(f"\n⚠️ Wrap-up workflow did not submit any work")
                    
            except Exception as e:
                self.logger.error(
                    f"Wrap-up workflow failed: {str(e)}",
                    context={"date": date, "task_id": self.current_task.get('task_id')},
                    exception=e,
                    print_console=True
                )

        # Clean up task-level sandbox to prevent accumulation
        # This ensures sandbox is killed after each task/day, not just at program exit
        try:
            from livebench.tools.productivity.code_execution_sandbox import SessionSandbox
            session_sandbox = SessionSandbox.get_instance()
            if session_sandbox.sandbox:
                session_sandbox.cleanup()
                self.logger.terminal_print("🧹 Cleaned up task sandbox")
        except Exception as e:
            self.logger.warning(
                f"Failed to cleanup task sandbox: {str(e)}",
                context={"date": date},
                print_console=False
            )

        # End of day: save economic state
        self.economic_tracker.save_daily_state(
            date=date,
            work_income=self.daily_work_income,
            trading_profit=self.daily_trading_profit
        )
        
        # Clean up E2B sandbox for this session
        try:
            from livebench.tools.productivity.code_execution_sandbox import cleanup_session_sandbox
            cleanup_session_sandbox()
        except Exception as e:
            self.logger.warning(
                f"Failed to cleanup E2B sandbox: {str(e)}",
                context={"date": date},
                print_console=False
            )

        print(f"\n{'='*60}")
        print(f"📊 Daily Summary - {date}")
        print(f"   Balance: ${self.economic_tracker.get_balance():.2f}")
        print(f"   Daily Cost: ${self.economic_tracker.get_daily_cost():.2f}")
        print(f"   Work Income: ${self.daily_work_income:.2f}")
        print(f"   Trading P&L: ${self.daily_trading_profit:.2f}")
        print(f"   Status: {self.economic_tracker.get_survival_status()}")
        print(f"{'='*60}\n")

    async def run_date_range(self, init_date: str, end_date: str) -> None:
        """
        Run simulation for date range

        Args:
            init_date: Start date
            end_date: End date
        """
        print(f"\n🎮 Starting LiveBench Simulation")
        print(f"   Agent: {self.signature}")
        print(f"   Model: {self.basemodel}")
        print(f"   Date Range: {init_date} to {end_date}")
        print(f"   Starting Balance: ${self.economic_tracker.initial_balance:.2f}\n")

        from datetime import datetime as dt, timedelta

        current_date = dt.strptime(init_date, "%Y-%m-%d")
        end = dt.strptime(end_date, "%Y-%m-%d")

        day_count = 0
        while current_date <= end:
            if current_date.weekday() < 5:  # Weekdays only
                day_count += 1
                date_str = current_date.strftime("%Y-%m-%d")

                result = await self.run_daily_session(date_str)

                # Check if no tasks available
                if result == "NO_TASKS_AVAILABLE":
                    print(f"\n🛑 SIMULATION ENDED - No more tasks available on {date_str}")
                    print(f"   Completed: {day_count} days")
                    print(f"   All available tasks have been assigned")
                    break

                # Check bankruptcy
                if self.economic_tracker.is_bankrupt():
                    print(f"\n💀 GAME OVER - Agent {self.signature} went bankrupt on {date_str}")
                    print(f"   Survived: {day_count} days")
                    break

            current_date += timedelta(days=1)

        # Final summary
        self._print_final_summary(day_count)

    def _print_final_summary(self, days_survived: int) -> None:
        """Print final simulation summary"""
        summary = self.economic_tracker.get_summary()

        print(f"\n{'='*60}")
        print(f"🏁 FINAL SUMMARY - {self.signature}")
        print(f"{'='*60}")
        print(f"   Days Survived: {days_survived}")
        print(f"   Final Balance: ${summary['balance']:.2f}")
        print(f"   Net Worth: ${summary['net_worth']:.2f}")
        print(f"   Total Token Cost: ${summary['total_token_cost']:.2f}")
        print(f"   Total Work Income: ${self.economic_tracker.total_work_income:.2f}")
        print(f"   Total Trading P&L: ${self.economic_tracker.total_trading_profit:.2f}")
        print(f"   Final Status: {summary['survival_status'].upper()}")
        print(f"{'='*60}\n")

    def __str__(self) -> str:
        return f"LiveAgent(signature='{self.signature}', model='{self.basemodel}')"
