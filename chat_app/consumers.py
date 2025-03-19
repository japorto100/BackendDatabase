import json
import asyncio
import uuid
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from .models import ChatSession, Message
from models_app.ai_models import AIModelManager
from django.utils import timezone
from models_app.mention_processor import MentionProcessor
from asgiref.sync import sync_to_async

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        
        if not self.user.is_authenticated:
            await self.close(code=4003)
            return
            
        await self.accept()
        
    async def disconnect(self, close_code):
        pass
        
    async def receive(self, text_data):
        data = json.loads(text_data)
        message_type = data.get('type', 'message')
        
        if message_type == 'typing':
            # User is typing, broadcast to channel group
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'typing_status',
                    'is_typing': data.get('is_typing', True),
                    'user_id': self.user.id,
                    'timestamp': timezone.now().isoformat()
                }
            )
        elif message_type == 'message':
            await self.handle_message(data)
            
    async def handle_message(self, data):
        message_id = data.get('message_id')
        session_id = data.get('session_id')
        content = data.get('content')
        model_id = data.get('model_id')
        mentions = data.get('mentions', [])
        
        if not all([message_id, session_id, content]):
            await self.send_error(message_id, "Missing required fields")
            return
            
        try:
            # Generiere eine query_id für die Anfrage
            query_id = uuid.uuid4()
            
            # Save user message
            user_message = await self.save_message(session_id, 'user', content)
            
            # Get AI model manager
            ai_model_manager = AIModelManager()
            
            # Get model info
            model_info = ai_model_manager.get_model_info(model_id)
            if not model_info:
                model_id = ai_model_manager.default_model
                model_info = ai_model_manager.get_model_info(model_id)
                
            # Get message history
            messages = await self.get_message_history(session_id)
            
            # NEUE FUNKTIONALITÄT: Verarbeite @-Mentions, falls vorhanden
            context_text = ""
            mention_sources = []
            if mentions:
                mention_processor = MentionProcessor()
                
                # Verarbeite Mentions und erhalte Kontext und Quellen
                context, sources, _ = await sync_to_async(mention_processor.process_mentions)(
                    mentions, str(query_id)
                )
                
                if context:
                    context_text = mention_processor.format_context(context)
                    mention_sources = sources
            
            # Erweitere die Nachrichten-Historie mit dem Kontext aus Mentions
            if context_text:
                # Füge den Kontext zu den Systemnachrichten hinzu
                system_msg = {
                    'role': 'system',
                    'content': f"Der Benutzer bezieht sich auf folgende Dokumente:\n\n{context_text}"
                }
                messages.insert(0, system_msg)
            
            # Generate AI response with streaming
            chunks = []  # Initialize empty list to collect chunks
            async for chunk in ai_model_manager.generate_streaming_response(model_id, messages):
                chunks.append(chunk)  # Add each chunk to our collection
                await self.send_chunk(message_id, chunk)
                
            # Save complete AI response
            complete_response = "".join(chunks)
            assistant_message = await self.save_message(session_id, 'assistant', complete_response)
            
            # Send completion message with zusätzlichen mention_sources
            await self.send_complete(message_id, {
                'user_message_id': str(user_message.id),
                'assistant_message_id': str(assistant_message.id),
                'model_used': model_id,
                'sources': mention_sources  # Füge die Quellen aus Mentions hinzu
            })
            
        except Exception as e:
            await self.send_error(message_id, str(e))
            
    async def send_chunk(self, message_id, chunk):
        await self.send(text_data=json.dumps({
            'type': 'chunk',
            'message_id': message_id,
            'content': chunk
        }))
        
    async def send_complete(self, message_id, data):
        await self.send(text_data=json.dumps({
            'type': 'complete',
            'message_id': message_id,
            **data
        }))
        
    async def send_error(self, message_id, error):
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message_id': message_id,
            'error': error
        }))
        
    @database_sync_to_async
    def save_message(self, session_id, role, content):
        session = ChatSession.objects.get(id=session_id, user=self.user)
        message = Message.objects.create(
            session=session,
            role=role,
            content=content
        )
        return message
        
    @database_sync_to_async
    def get_message_history(self, session_id):
        session = ChatSession.objects.get(id=session_id, user=self.user)
        messages = list(Message.objects.filter(session=session).order_by('timestamp'))
        return [{'role': msg.role, 'content': msg.content} for msg in messages] 

    async def typing_status(self, event):
        """
        Send typing status to WebSocket
        """
        message = {
            'type': 'typing_status',
            'is_typing': event['is_typing'],
            'user_id': event['user_id'],
            'timestamp': event['timestamp']
        }
        
        # Send typing status to WebSocket
        await self.send(text_data=json.dumps(message)) 