"""
Dialog Flow Engine for managing conversation flows using YAML-based definitions.
Supports dynamic branching, slot filling, and context-aware responses.
"""

import asyncio
import logging
import yaml
import re
import json
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import uuid

from .state_manager import ConversationSession, ConversationContext, conversation_state_manager
from ai.llm.llm_service import llm_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class FlowNodeType(Enum):
    """Types of flow nodes."""
    MESSAGE = "message"
    QUESTION = "question"
    CONDITION = "condition"
    ACTION = "action"
    SLOT_FILL = "slot_fill"
    TRANSFER = "transfer"
    END = "end"
    WEBHOOK = "webhook"
    LLM_RESPONSE = "llm_response"


class SlotType(Enum):
    """Types of slots for information gathering."""
    TEXT = "text"
    NUMBER = "number"
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    TIME = "time"
    BOOLEAN = "boolean"
    CHOICE = "choice"


@dataclass
class Slot:
    """Slot definition for information gathering."""
    name: str
    type: SlotType
    prompt: str
    required: bool = True
    validation_pattern: Optional[str] = None
    choices: Optional[List[str]] = None
    retry_prompt: Optional[str] = None
    max_retries: int = 3
    default_value: Optional[Any] = None
    
    def validate(self, value: str) -> tuple[bool, Any]:
        """Validate slot value and return (is_valid, processed_value)."""
        if not value and self.required:
            return False, None
        
        if not value and self.default_value is not None:
            return True, self.default_value
        
        try:
            if self.type == SlotType.TEXT:
                if self.validation_pattern:
                    if not re.match(self.validation_pattern, value):
                        return False, None
                return True, value
            
            elif self.type == SlotType.NUMBER:
                return True, float(value)
            
            elif self.type == SlotType.EMAIL:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if re.match(email_pattern, value):
                    return True, value
                return False, None
            
            elif self.type == SlotType.PHONE:
                # Simple phone validation
                phone_pattern = r'^[\+]?[1-9][\d]{0,15}$'
                clean_phone = re.sub(r'[^\d+]', '', value)
                if re.match(phone_pattern, clean_phone):
                    return True, clean_phone
                return False, None
            
            elif self.type == SlotType.DATE:
                # Basic date validation - would need more sophisticated parsing
                return True, value
            
            elif self.type == SlotType.TIME:
                # Basic time validation
                time_pattern = r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$'
                if re.match(time_pattern, value):
                    return True, value
                return False, None
            
            elif self.type == SlotType.BOOLEAN:
                lower_value = value.lower()
                if lower_value in ['yes', 'y', 'true', '1', 'ok', 'sure']:
                    return True, True
                elif lower_value in ['no', 'n', 'false', '0', 'nope']:
                    return True, False
                return False, None
            
            elif self.type == SlotType.CHOICE:
                if self.choices:
                    # Fuzzy matching for choices
                    value_lower = value.lower()
                    for choice in self.choices:
                        if value_lower in choice.lower() or choice.lower() in value_lower:
                            return True, choice
                return False, None
            
            return True, value
            
        except Exception:
            return False, None


@dataclass
class FlowNode:
    """A node in the conversation flow."""
    id: str
    type: FlowNodeType
    name: Optional[str] = None
    
    # Message/Question content
    message: Optional[str] = None
    messages: Optional[List[str]] = None  # For random selection
    
    # Conditions
    condition: Optional[str] = None
    condition_variable: Optional[str] = None
    condition_operator: Optional[str] = None
    condition_value: Optional[Any] = None
    
    # Actions
    action: Optional[str] = None
    action_params: Optional[Dict[str, Any]] = None
    
    # Slot filling
    slot: Optional[Slot] = None
    slots: Optional[List[Slot]] = None
    
    # Navigation
    next_node: Optional[str] = None
    branches: Optional[Dict[str, str]] = None  # condition -> node_id
    
    # LLM integration
    llm_prompt: Optional[str] = None
    llm_system_prompt: Optional[str] = None
    use_context: bool = True
    
    # Webhook
    webhook_url: Optional[str] = None
    webhook_method: str = "POST"
    webhook_params: Optional[Dict[str, Any]] = None
    
    # Transfer
    transfer_to: Optional[str] = None
    transfer_reason: Optional[str] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationFlow:
    """Complete conversation flow definition."""
    id: str
    name: str
    description: Optional[str]
    start_node: str
    nodes: Dict[str, FlowNode]
    global_variables: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class FlowExecutionContext:
    """Context for flow execution."""
    
    def __init__(self, session_id: str, flow: ConversationFlow):
        self.session_id = session_id
        self.flow = flow
        self.current_node_id = flow.start_node
        self.variables: Dict[str, Any] = flow.global_variables.copy() if flow.global_variables else {}
        self.collected_slots: Dict[str, Any] = {}
        self.execution_history: List[str] = []
        self.retry_counts: Dict[str, int] = {}
        self.started_at = datetime.utcnow()
        
    def set_variable(self, name: str, value: Any):
        """Set a flow variable."""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a flow variable."""
        return self.variables.get(name, default)
    
    def set_slot_value(self, slot_name: str, value: Any):
        """Set a collected slot value."""
        self.collected_slots[slot_name] = value
    
    def get_slot_value(self, slot_name: str, default: Any = None) -> Any:
        """Get a collected slot value."""
        return self.collected_slots.get(slot_name, default)
    
    def navigate_to(self, node_id: str):
        """Navigate to a specific node."""
        self.execution_history.append(self.current_node_id)
        self.current_node_id = node_id
    
    def increment_retry(self, key: str) -> int:
        """Increment retry count for a key."""
        self.retry_counts[key] = self.retry_counts.get(key, 0) + 1
        return self.retry_counts[key]


class FlowLoader:
    """Loads conversation flows from YAML files."""
    
    @staticmethod
    def load_flow_from_file(file_path: str) -> ConversationFlow:
        """Load a conversation flow from YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                flow_data = yaml.safe_load(file)
            
            return FlowLoader.load_flow_from_dict(flow_data)
            
        except Exception as e:
            logger.error(f"Error loading flow from {file_path}: {e}")
            raise
    
    @staticmethod
    def load_flow_from_dict(flow_data: Dict[str, Any]) -> ConversationFlow:
        """Load a conversation flow from dictionary."""
        try:
            # Parse nodes
            nodes = {}
            for node_id, node_data in flow_data.get('nodes', {}).items():
                node = FlowLoader._parse_node(node_id, node_data)
                nodes[node_id] = node
            
            # Create flow
            flow = ConversationFlow(
                id=flow_data['id'],
                name=flow_data['name'],
                description=flow_data.get('description'),
                start_node=flow_data['start_node'],
                nodes=nodes,
                global_variables=flow_data.get('global_variables'),
                metadata=flow_data.get('metadata')
            )
            
            return flow
            
        except Exception as e:
            logger.error(f"Error parsing flow data: {e}")
            raise
    
    @staticmethod
    def _parse_node(node_id: str, node_data: Dict[str, Any]) -> FlowNode:
        """Parse a single flow node."""
        node_type = FlowNodeType(node_data['type'])
        
        # Parse slot if present
        slot = None
        if 'slot' in node_data:
            slot_data = node_data['slot']
            slot = Slot(
                name=slot_data['name'],
                type=SlotType(slot_data['type']),
                prompt=slot_data['prompt'],
                required=slot_data.get('required', True),
                validation_pattern=slot_data.get('validation_pattern'),
                choices=slot_data.get('choices'),
                retry_prompt=slot_data.get('retry_prompt'),
                max_retries=slot_data.get('max_retries', 3),
                default_value=slot_data.get('default_value')
            )
        
        # Parse multiple slots if present
        slots = None
        if 'slots' in node_data:
            slots = []
            for slot_data in node_data['slots']:
                slot_obj = Slot(
                    name=slot_data['name'],
                    type=SlotType(slot_data['type']),
                    prompt=slot_data['prompt'],
                    required=slot_data.get('required', True),
                    validation_pattern=slot_data.get('validation_pattern'),
                    choices=slot_data.get('choices'),
                    retry_prompt=slot_data.get('retry_prompt'),
                    max_retries=slot_data.get('max_retries', 3),
                    default_value=slot_data.get('default_value')
                )
                slots.append(slot_obj)
        
        return FlowNode(
            id=node_id,
            type=node_type,
            name=node_data.get('name'),
            message=node_data.get('message'),
            messages=node_data.get('messages'),
            condition=node_data.get('condition'),
            condition_variable=node_data.get('condition_variable'),
            condition_operator=node_data.get('condition_operator'),
            condition_value=node_data.get('condition_value'),
            action=node_data.get('action'),
            action_params=node_data.get('action_params'),
            slot=slot,
            slots=slots,
            next_node=node_data.get('next_node'),
            branches=node_data.get('branches'),
            llm_prompt=node_data.get('llm_prompt'),
            llm_system_prompt=node_data.get('llm_system_prompt'),
            use_context=node_data.get('use_context', True),
            webhook_url=node_data.get('webhook_url'),
            webhook_method=node_data.get('webhook_method', 'POST'),
            webhook_params=node_data.get('webhook_params'),
            transfer_to=node_data.get('transfer_to'),
            transfer_reason=node_data.get('transfer_reason'),
            metadata=node_data.get('metadata')
        )


class DialogFlowEngine:
    """Main dialog flow engine for conversation management."""
    
    def __init__(self):
        self.flows: Dict[str, ConversationFlow] = {}
        self.execution_contexts: Dict[str, FlowExecutionContext] = {}
        self.flow_directory = Path("ai/conversation/flows")
        
        # Action handlers
        self.action_handlers: Dict[str, Callable] = {}
        
        # Built-in actions
        self._register_builtin_actions()
    
    def _register_builtin_actions(self):
        """Register built-in action handlers."""
        self.action_handlers.update({
            'set_variable': self._action_set_variable,
            'log_message': self._action_log_message,
            'end_conversation': self._action_end_conversation,
            'escalate_to_human': self._action_escalate_to_human,
        })
    
    async def initialize(self) -> bool:
        """Initialize the dialog flow engine."""
        try:
            # Create flows directory if it doesn't exist
            self.flow_directory.mkdir(parents=True, exist_ok=True)
            
            # Load default flows
            await self._load_default_flows()
            
            # Load flows from directory
            await self.load_flows_from_directory()
            
            logger.info(f"Dialog flow engine initialized with {len(self.flows)} flows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dialog flow engine: {e}")
            return False
    
    async def _load_default_flows(self):
        """Load default conversation flows."""
        # Legal consultation flow
        legal_flow = self._create_legal_consultation_flow()
        self.flows[legal_flow.id] = legal_flow
        
        # Appointment booking flow
        appointment_flow = self._create_appointment_booking_flow()
        self.flows[appointment_flow.id] = appointment_flow
    
    def _create_legal_consultation_flow(self) -> ConversationFlow:
        """Create a legal consultation flow."""
        nodes = {
            'greeting': FlowNode(
                id='greeting',
                type=FlowNodeType.MESSAGE,
                message="Hello! I'm your AI legal assistant. I'm here to help you with your legal questions and connect you with the right resources. How can I assist you today?",
                next_node='get_issue_type'
            ),
            'get_issue_type': FlowNode(
                id='get_issue_type',
                type=FlowNodeType.SLOT_FILL,
                slot=Slot(
                    name='issue_type',
                    type=SlotType.CHOICE,
                    prompt="What type of legal matter do you need help with?",
                    choices=['Contract Review', 'Personal Injury', 'Family Law', 'Business Law', 'Real Estate', 'Other'],
                    required=True
                ),
                next_node='get_details'
            ),
            'get_details': FlowNode(
                id='get_details',
                type=FlowNodeType.SLOT_FILL,
                slot=Slot(
                    name='issue_details',
                    type=SlotType.TEXT,
                    prompt="Can you provide more details about your legal issue? Please be as specific as possible.",
                    required=True
                ),
                next_node='analyze_urgency'
            ),
            'analyze_urgency': FlowNode(
                id='analyze_urgency',
                type=FlowNodeType.LLM_RESPONSE,
                llm_prompt="Based on the legal issue type '{issue_type}' and details '{issue_details}', assess the urgency and provide initial guidance. Is this urgent?",
                llm_system_prompt="You are a legal assistant. Analyze the urgency of the legal matter and provide helpful initial guidance.",
                next_node='check_urgency'
            ),
            'check_urgency': FlowNode(
                id='check_urgency',
                type=FlowNodeType.QUESTION,
                message="Based on your description, would you say this matter is urgent and needs immediate attention?",
                branches={
                    'yes': 'urgent_path',
                    'no': 'normal_path'
                }
            ),
            'urgent_path': FlowNode(
                id='urgent_path',
                type=FlowNodeType.MESSAGE,
                message="I understand this is urgent. Let me connect you with an attorney right away. Please hold while I transfer your call.",
                next_node='transfer_to_attorney'
            ),
            'normal_path': FlowNode(
                id='normal_path',
                type=FlowNodeType.MESSAGE,
                message="Thank you for the information. Let me see how I can best assist you. Would you like me to schedule a consultation or would you prefer some initial guidance?",
                next_node='consultation_or_guidance'
            ),
            'consultation_or_guidance': FlowNode(
                id='consultation_or_guidance',
                type=FlowNodeType.QUESTION,
                message="Would you like to: 1) Schedule a consultation with an attorney, or 2) Get some initial guidance on your matter?",
                branches={
                    '1': 'schedule_consultation',
                    '2': 'provide_guidance',
                    'schedule': 'schedule_consultation',
                    'guidance': 'provide_guidance'
                }
            ),
            'schedule_consultation': FlowNode(
                id='schedule_consultation',
                type=FlowNodeType.ACTION,
                action='transfer_to_flow',
                action_params={'flow_id': 'appointment_booking'},
                message="I'll help you schedule a consultation. Let me gather some information."
            ),
            'provide_guidance': FlowNode(
                id='provide_guidance',
                type=FlowNodeType.LLM_RESPONSE,
                llm_prompt="Provide helpful legal guidance for the issue type '{issue_type}' with details '{issue_details}'. Include relevant laws, next steps, and when they should consult an attorney.",
                llm_system_prompt="You are a knowledgeable legal assistant. Provide helpful guidance while making it clear you're not providing legal advice and they should consult with an attorney for specific legal advice.",
                next_node='offer_consultation'
            ),
            'offer_consultation': FlowNode(
                id='offer_consultation',
                type=FlowNodeType.QUESTION,
                message="I hope that information was helpful. Would you like to schedule a consultation with one of our attorneys to discuss your matter in more detail?",
                branches={
                    'yes': 'schedule_consultation',
                    'no': 'end_helpful'
                }
            ),
            'transfer_to_attorney': FlowNode(
                id='transfer_to_attorney',
                type=FlowNodeType.TRANSFER,
                transfer_to='attorney_queue',
                transfer_reason='urgent_legal_matter'
            ),
            'end_helpful': FlowNode(
                id='end_helpful',
                type=FlowNodeType.END,
                message="Thank you for calling. If you need any assistance in the future, please don't hesitate to contact us. Have a great day!"
            )
        }
        
        return ConversationFlow(
            id='legal_consultation',
            name='Legal Consultation Flow',
            description='Flow for handling legal consultations and questions',
            start_node='greeting',
            nodes=nodes
        )
    
    def _create_appointment_booking_flow(self) -> ConversationFlow:
        """Create an appointment booking flow."""
        nodes = {
            'start': FlowNode(
                id='start',
                type=FlowNodeType.MESSAGE,
                message="I'd be happy to help you schedule an appointment. Let me gather some information.",
                next_node='get_name'
            ),
            'get_name': FlowNode(
                id='get_name',
                type=FlowNodeType.SLOT_FILL,
                slot=Slot(
                    name='client_name',
                    type=SlotType.TEXT,
                    prompt="Could you please provide your full name?",
                    required=True
                ),
                next_node='get_phone'
            ),
            'get_phone': FlowNode(
                id='get_phone',
                type=FlowNodeType.SLOT_FILL,
                slot=Slot(
                    name='phone_number',
                    type=SlotType.PHONE,
                    prompt="What's the best phone number to reach you at?",
                    required=True
                ),
                next_node='get_email'
            ),
            'get_email': FlowNode(
                id='get_email',
                type=FlowNodeType.SLOT_FILL,
                slot=Slot(
                    name='email',
                    type=SlotType.EMAIL,
                    prompt="And what's your email address?",
                    required=False
                ),
                next_node='get_preferred_time'
            ),
            'get_preferred_time': FlowNode(
                id='get_preferred_time',
                type=FlowNodeType.SLOT_FILL,
                slot=Slot(
                    name='preferred_time',
                    type=SlotType.CHOICE,
                    prompt="When would you prefer to meet? Morning, afternoon, or evening?",
                    choices=['morning', 'afternoon', 'evening'],
                    required=True
                ),
                next_node='confirm_appointment'
            ),
            'confirm_appointment': FlowNode(
                id='confirm_appointment',
                type=FlowNodeType.MESSAGE,
                message="Perfect! I have your information: {client_name}, phone: {phone_number}, email: {email}, preferred time: {preferred_time}. Our scheduler will contact you within 24 hours to confirm your appointment time.",
                next_node='end'
            ),
            'end': FlowNode(
                id='end',
                type=FlowNodeType.END,
                message="Thank you for choosing our firm. We look forward to speaking with you soon!"
            )
        }
        
        return ConversationFlow(
            id='appointment_booking',
            name='Appointment Booking Flow',
            description='Flow for booking legal consultations',
            start_node='start',
            nodes=nodes
        )
    
    async def load_flows_from_directory(self):
        """Load all flows from the flows directory."""
        try:
            if not self.flow_directory.exists():
                logger.warning(f"Flows directory {self.flow_directory} does not exist")
                return
            
            for yaml_file in self.flow_directory.glob("*.yaml"):
                try:
                    flow = FlowLoader.load_flow_from_file(str(yaml_file))
                    self.flows[flow.id] = flow
                    logger.info(f"Loaded flow: {flow.id} from {yaml_file}")
                except Exception as e:
                    logger.error(f"Error loading flow from {yaml_file}: {e}")
            
        except Exception as e:
            logger.error(f"Error loading flows from directory: {e}")
    
    def register_action_handler(self, action_name: str, handler: Callable):
        """Register a custom action handler."""
        self.action_handlers[action_name] = handler
    
    async def start_flow(self, session_id: str, flow_id: str) -> Optional[str]:
        """Start a conversation flow for a session."""
        try:
            if flow_id not in self.flows:
                logger.error(f"Flow {flow_id} not found")
                return None
            
            flow = self.flows[flow_id]
            context = FlowExecutionContext(session_id, flow)
            self.execution_contexts[session_id] = context
            
            # Execute the start node
            response = await self._execute_node(context, flow.start_node)
            
            logger.info(f"Started flow {flow_id} for session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error starting flow: {e}")
            return None
    
    async def process_user_input(self, session_id: str, user_input: str) -> Optional[str]:
        """Process user input in the context of the current flow."""
        try:
            if session_id not in self.execution_contexts:
                logger.error(f"No active flow for session {session_id}")
                return None
            
            context = self.execution_contexts[session_id]
            current_node = context.flow.nodes[context.current_node_id]
            
            # Handle different node types
            if current_node.type == FlowNodeType.SLOT_FILL:
                return await self._handle_slot_fill(context, current_node, user_input)
            elif current_node.type == FlowNodeType.QUESTION:
                return await self._handle_question(context, current_node, user_input)
            else:
                # For other node types, process input and continue
                return await self._continue_flow(context)
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error. Could you please try again?"
    
    async def _execute_node(self, context: FlowExecutionContext, node_id: str) -> Optional[str]:
        """Execute a specific node in the flow."""
        try:
            if node_id not in context.flow.nodes:
                logger.error(f"Node {node_id} not found in flow")
                return None
            
            node = context.flow.nodes[node_id]
            context.current_node_id = node_id
            
            if node.type == FlowNodeType.MESSAGE:
                return await self._handle_message(context, node)
            elif node.type == FlowNodeType.QUESTION:
                return await self._handle_question_prompt(context, node)
            elif node.type == FlowNodeType.SLOT_FILL:
                return await self._handle_slot_fill_prompt(context, node)
            elif node.type == FlowNodeType.CONDITION:
                return await self._handle_condition(context, node)
            elif node.type == FlowNodeType.ACTION:
                return await self._handle_action(context, node)
            elif node.type == FlowNodeType.LLM_RESPONSE:
                return await self._handle_llm_response(context, node)
            elif node.type == FlowNodeType.TRANSFER:
                return await self._handle_transfer(context, node)
            elif node.type == FlowNodeType.END:
                return await self._handle_end(context, node)
            else:
                logger.warning(f"Unknown node type: {node.type}")
                return await self._continue_flow(context)
            
        except Exception as e:
            logger.error(f"Error executing node {node_id}: {e}")
            return "I'm sorry, I encountered an error. Let me try to continue."
    
    async def _handle_message(self, context: FlowExecutionContext, node: FlowNode) -> Optional[str]:
        """Handle a message node."""
        message = self._format_message(node.message, context)
        
        if node.next_node:
            # If there's a next node, execute it immediately
            context.navigate_to(node.next_node)
            next_response = await self._execute_node(context, node.next_node)
            if next_response:
                return f"{message}\n\n{next_response}"
        
        return message
    
    async def _handle_question_prompt(self, context: FlowExecutionContext, node: FlowNode) -> Optional[str]:
        """Handle the prompt for a question node."""
        return self._format_message(node.message, context)
    
    async def _handle_question(self, context: FlowExecutionContext, node: FlowNode, user_input: str) -> Optional[str]:
        """Handle user response to a question node."""
        user_input_lower = user_input.lower().strip()
        
        # Check branches
        if node.branches:
            for branch_key, next_node in node.branches.items():
                if branch_key.lower() in user_input_lower or user_input_lower in branch_key.lower():
                    context.navigate_to(next_node)
                    return await self._execute_node(context, next_node)
        
        # If no branch matches and there's a default next node
        if node.next_node:
            context.navigate_to(node.next_node)
            return await self._execute_node(context, node.next_node)
        
        return "I didn't understand your response. Could you please try again?"
    
    async def _handle_slot_fill_prompt(self, context: FlowExecutionContext, node: FlowNode) -> Optional[str]:
        """Handle the prompt for a slot fill node."""
        if node.slot:
            return node.slot.prompt
        return "Please provide the requested information."
    
    async def _handle_slot_fill(self, context: FlowExecutionContext, node: FlowNode, user_input: str) -> Optional[str]:
        """Handle slot filling."""
        if not node.slot:
            return await self._continue_flow(context)
        
        slot = node.slot
        
        # Validate the input
        is_valid, processed_value = slot.validate(user_input)
        
        if is_valid:
            # Store the slot value
            context.set_slot_value(slot.name, processed_value)
            context.set_variable(slot.name, processed_value)
            
            # Continue to next node
            if node.next_node:
                context.navigate_to(node.next_node)
                return await self._execute_node(context, node.next_node)
            else:
                return await self._continue_flow(context)
        else:
            # Invalid input, ask again
            retry_count = context.increment_retry(f"slot_{slot.name}")
            
            if retry_count >= slot.max_retries:
                # Max retries reached, either use default or error
                if slot.default_value is not None:
                    context.set_slot_value(slot.name, slot.default_value)
                    context.set_variable(slot.name, slot.default_value)
                    if node.next_node:
                        context.navigate_to(node.next_node)
                        return await self._execute_node(context, node.next_node)
                else:
                    return "I'm sorry, I couldn't get valid information. Let me transfer you to a human agent."
            
            retry_message = slot.retry_prompt or f"I'm sorry, that doesn't seem to be a valid {slot.type.value}. {slot.prompt}"
            return retry_message
    
    async def _handle_condition(self, context: FlowExecutionContext, node: FlowNode) -> Optional[str]:
        """Handle a condition node."""
        # Simple condition evaluation
        if node.condition_variable and node.condition_operator and node.condition_value is not None:
            variable_value = context.get_variable(node.condition_variable)
            
            if node.condition_operator == "equals" and variable_value == node.condition_value:
                condition_met = True
            elif node.condition_operator == "not_equals" and variable_value != node.condition_value:
                condition_met = True
            elif node.condition_operator == "greater_than" and variable_value > node.condition_value:
                condition_met = True
            elif node.condition_operator == "less_than" and variable_value < node.condition_value:
                condition_met = True
            else:
                condition_met = False
            
            # Navigate based on condition
            if condition_met and node.branches and "true" in node.branches:
                context.navigate_to(node.branches["true"])
                return await self._execute_node(context, node.branches["true"])
            elif not condition_met and node.branches and "false" in node.branches:
                context.navigate_to(node.branches["false"])
                return await self._execute_node(context, node.branches["false"])
        
        return await self._continue_flow(context)
    
    async def _handle_action(self, context: FlowExecutionContext, node: FlowNode) -> Optional[str]:
        """Handle an action node."""
        if node.action and node.action in self.action_handlers:
            try:
                handler = self.action_handlers[node.action]
                result = await handler(context, node.action_params or {})
                
                # Continue to next node
                if node.next_node:
                    context.navigate_to(node.next_node)
                    return await self._execute_node(context, node.next_node)
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing action {node.action}: {e}")
        
        return await self._continue_flow(context)
    
    async def _handle_llm_response(self, context: FlowExecutionContext, node: FlowNode) -> Optional[str]:
        """Handle an LLM response node."""
        try:
            # Format the prompt with context variables
            prompt = self._format_message(node.llm_prompt, context)
            system_prompt = node.llm_system_prompt
            
            # Add conversation context if requested
            conversation_id = context.session_id if node.use_context else None
            
            # Generate response
            response = ""
            async for chunk in llm_service.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_id=conversation_id,
                stream=False
            ):
                response += chunk
            
            # Continue to next node
            if node.next_node:
                # Add a small delay before continuing to make conversation feel natural
                await asyncio.sleep(0.5)
                context.navigate_to(node.next_node)
                next_response = await self._execute_node(context, node.next_node)
                if next_response:
                    return f"{response}\n\n{next_response}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I'm sorry, I'm having trouble processing that right now. How else can I help you?"
    
    async def _handle_transfer(self, context: FlowExecutionContext, node: FlowNode) -> Optional[str]:
        """Handle a transfer node."""
        # In a real implementation, this would trigger a call transfer
        message = f"I'm transferring you to {node.transfer_to}."
        if node.transfer_reason:
            message += f" Reason: {node.transfer_reason}"
        
        # End the flow context
        if context.session_id in self.execution_contexts:
            del self.execution_contexts[context.session_id]
        
        return message
    
    async def _handle_end(self, context: FlowExecutionContext, node: FlowNode) -> Optional[str]:
        """Handle an end node."""
        message = node.message or "Thank you for your time. Have a great day!"
        message = self._format_message(message, context)
        
        # Clean up the execution context
        if context.session_id in self.execution_contexts:
            del self.execution_contexts[context.session_id]
        
        return message
    
    def _format_message(self, message: str, context: FlowExecutionContext) -> str:
        """Format message with context variables."""
        if not message:
            return ""
        
        try:
            # Replace variables in the format {variable_name}
            for var_name, var_value in context.variables.items():
                message = message.replace(f"{{{var_name}}}", str(var_value))
            
            for slot_name, slot_value in context.collected_slots.items():
                message = message.replace(f"{{{slot_name}}}", str(slot_value))
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting message: {e}")
            return message
    
    async def _continue_flow(self, context: FlowExecutionContext) -> Optional[str]:
        """Continue flow execution to the next node."""
        current_node = context.flow.nodes[context.current_node_id]
        
        if current_node.next_node:
            context.navigate_to(current_node.next_node)
            return await self._execute_node(context, current_node.next_node)
        
        return None
    
    # Built-in action handlers
    async def _action_set_variable(self, context: FlowExecutionContext, params: Dict[str, Any]) -> str:
        """Set a variable action."""
        if 'name' in params and 'value' in params:
            context.set_variable(params['name'], params['value'])
        return ""
    
    async def _action_log_message(self, context: FlowExecutionContext, params: Dict[str, Any]) -> str:
        """Log a message action."""
        message = params.get('message', 'Action executed')
        logger.info(f"Flow action log: {message}")
        return ""
    
    async def _action_end_conversation(self, context: FlowExecutionContext, params: Dict[str, Any]) -> str:
        """End conversation action."""
        await conversation_state_manager.end_session(context.session_id)
        return params.get('message', 'Thank you for your time.')
    
    async def _action_escalate_to_human(self, context: FlowExecutionContext, params: Dict[str, Any]) -> str:
        """Escalate to human action."""
        reason = params.get('reason', 'User requested human assistance')
        return f"I'm connecting you with a human agent. {reason}"
    
    def get_active_flows(self) -> List[str]:
        """Get list of session IDs with active flows."""
        return list(self.execution_contexts.keys())
    
    def get_flow_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a flow execution."""
        if session_id not in self.execution_contexts:
            return None
        
        context = self.execution_contexts[session_id]
        return {
            'session_id': session_id,
            'flow_id': context.flow.id,
            'current_node': context.current_node_id,
            'variables': context.variables,
            'collected_slots': context.collected_slots,
            'execution_history': context.execution_history,
            'started_at': context.started_at.isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of dialog flow engine."""
        try:
            return {
                "healthy": True,
                "loaded_flows": len(self.flows),
                "active_executions": len(self.execution_contexts),
                "available_flows": list(self.flows.keys()),
                "action_handlers": len(self.action_handlers),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dialog flow health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global dialog flow engine instance
dialog_flow_engine = DialogFlowEngine()