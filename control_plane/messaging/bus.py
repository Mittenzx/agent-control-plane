"""
Message bus for inter-agent communication.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from collections import defaultdict
from datetime import datetime

from ..core.interfaces import Message, MessageType, Agent

logger = logging.getLogger(__name__)


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self):
        self._subscriptions: Dict[str, Set[Callable[[Message], None]]] = defaultdict(set)
        self._broadcast_subscriptions: Set[Callable[[Message], None]] = set()
        self._message_history: List[Message] = []
        self._max_history = 10000
        self._agent_addresses: Dict[str, asyncio.Queue] = {}  # agent_id -> message queue
    
    def subscribe(self, agent_id: str, handler: Callable[[Message], None]) -> None:
        """Subscribe an agent to messages addressed to it."""
        self._subscriptions[agent_id].add(handler)
        logger.debug(f"Agent {agent_id} subscribed to messages")
    
    def unsubscribe(self, agent_id: str, handler: Callable[[Message], None]) -> None:
        """Unsubscribe an agent."""
        self._subscriptions[agent_id].discard(handler)
        if not self._subscriptions[agent_id]:
            self._subscriptions.pop(agent_id, None)
        logger.debug(f"Agent {agent_id} unsubscribed from messages")
    
    def subscribe_broadcast(self, handler: Callable[[Message], None]) -> None:
        """Subscribe to all broadcast messages."""
        self._broadcast_subscriptions.add(handler)
    
    def unsubscribe_broadcast(self, handler: Callable[[Message], None]) -> None:
        """Unsubscribe from broadcasts."""
        self._broadcast_subscriptions.discard(handler)
    
    def register_agent_queue(self, agent_id: str, queue: asyncio.Queue) -> None:
        """Register an agent's message queue for direct delivery."""
        self._agent_addresses[agent_id] = queue
    
    def unregister_agent_queue(self, agent_id: str) -> None:
        """Unregister an agent's queue."""
        self._agent_addresses.pop(agent_id, None)
    
    async def send(self, message: Message) -> None:
        """Send a message to a specific recipient or broadcast."""
        # Store in history
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history:]
        
        if message.recipient_id:
            # Direct message
            await self._deliver_direct(message)
        else:
            # Broadcast
            await self._broadcast(message)
    
    async def _deliver_direct(self, message: Message) -> None:
        """Deliver message to specific recipient."""
        # Try queue delivery first
        queue = self._agent_addresses.get(message.recipient_id)
        if queue:
            try:
                queue.put_nowait(message)
                return
            except asyncio.QueueFull:
                logger.warning(f"Queue full for agent {message.recipient_id}")
        
        # Fallback to callback delivery
        handlers = self._subscriptions.get(message.recipient_id, set())
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Error delivering message to {message.recipient_id}: {e}")
    
    async def _broadcast(self, message: Message) -> None:
        """Broadcast message to all subscribers."""
        for handler in self._broadcast_subscriptions:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Error in broadcast handler: {e}")
    
    def get_history(self, limit: int = 100, sender_id: Optional[str] = None, 
                    recipient_id: Optional[str] = None, 
                    msg_type: Optional[MessageType] = None) -> List[Message]:
        """Get message history with optional filters."""
        messages = self._message_history
        
        if sender_id:
            messages = [m for m in messages if m.sender_id == sender_id]
        if recipient_id:
            messages = [m for m in messages if m.recipient_id == recipient_id]
        if msg_type:
            messages = [m for m in messages if m.type == msg_type]
        
        return messages[-limit:]
    
    def clear_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()


class RequestResponseBus(MessageBus):
    """Extended message bus with request/response pattern support."""
    
    def __init__(self):
        super().__init__()
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._request_timeout = 30.0  # seconds
    
    async def request(self, sender_id: str, recipient_id: str, 
                      payload: Dict[str, Any], 
                      msg_type: MessageType = MessageType.TASK_REQUEST,
                      timeout: Optional[float] = None) -> Message:
        """Send a request and wait for response."""
        import uuid
        correlation_id = str(uuid.uuid4())
        
        request = Message(
            type=msg_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload,
            correlation_id=correlation_id
        )
        
        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[correlation_id] = future
        
        # Send request
        await self.send(request)
        
        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=timeout or self._request_timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(correlation_id, None)
            raise TimeoutError(f"Request {correlation_id} timed out")
        except Exception as e:
            self._pending_requests.pop(correlation_id, None)
            raise
    
    async def respond(self, original_message: Message, response_payload: Dict[str, Any],
                      msg_type: MessageType = MessageType.TASK_RESPONSE) -> None:
        """Send a response to a request."""
        response = Message(
            type=msg_type,
            sender_id=original_message.recipient_id or "",
            recipient_id=original_message.sender_id,
            payload=response_payload,
            correlation_id=original_message.correlation_id
        )
        
        # Resolve pending request
        future = self._pending_requests.pop(original_message.correlation_id, None)
        if future and not future.done():
            future.set_result(response)
        
        # Also send normally for other subscribers
        await self.send(response)
    
    def handle_response(self, message: Message) -> bool:
        """Handle an incoming response message. Returns True if handled."""
        if not message.correlation_id:
            return False
        
        future = self._pending_requests.pop(message.correlation_id, None)
        if future and not future.done():
            future.set_result(message)
            return True
        return False


class EventBus:
    """Event bus for system-wide events."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._wildcard_subscribers: List[Callable] = []
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
    
    def subscribe_all(self, handler: Callable) -> None:
        """Subscribe to all events."""
        self._wildcard_subscribers.append(handler)
    
    def unsubscribe_all(self, handler: Callable) -> None:
        """Unsubscribe from all events."""
        if handler in self._wildcard_subscribers:
            self._wildcard_subscribers.remove(handler)
    
    async def emit(self, event_type: str, data: Any = None) -> None:
        """Emit an event to all subscribers."""
        # Specific subscribers
        for handler in self._subscribers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
        
        # Wildcard subscribers
        for handler in self._wildcard_subscribers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"Error in wildcard event handler: {e}")
    
    def emit_sync(self, event_type: str, data: Any = None) -> None:
        """Emit an event synchronously (fire and forget)."""
        asyncio.create_task(self.emit(event_type, data))


# System event types
class SystemEvents:
    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    TASK_SUBMITTED = "task.submitted"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"