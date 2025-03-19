export class StreamingResponseHandler {
    constructor(url) {
        this.url = url || `ws://${window.location.host}/ws/chat/`;
        this.socket = null;
        this.messageCallbacks = new Map();
        this.errorCallbacks = new Map();
        this.completeCallbacks = new Map();
    }

    connect() {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            this.socket = new WebSocket(this.url);
            
            this.socket.onopen = () => {
                console.log('WebSocket connection established');
                resolve();
            };
            
            this.socket.onclose = (event) => {
                console.log('WebSocket connection closed', event);
                // Attempt to reconnect after a delay
                setTimeout(() => this.connect(), 3000);
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    const { message_id, type, content, error } = data;
                    
                    if (type === 'error' && error) {
                        const callback = this.errorCallbacks.get(message_id);
                        if (callback) callback(error);
                    } 
                    else if (type === 'chunk') {
                        const callback = this.messageCallbacks.get(message_id);
                        if (callback) callback(content);
                    } 
                    else if (type === 'complete') {
                        const callback = this.completeCallbacks.get(message_id);
                        if (callback) callback(data);
                        
                        // Clean up callbacks
                        this.messageCallbacks.delete(message_id);
                        this.errorCallbacks.delete(message_id);
                        this.completeCallbacks.delete(message_id);
                    }
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };
        });
    }

    async sendMessage(sessionId, message, modelId) {
        await this.connect();
        
        const messageId = crypto.randomUUID();
        
        return new Promise((resolve, reject) => {
            // Set up callbacks
            this.messageCallbacks.set(messageId, (chunk) => {
                // This will be called for each chunk of the response
                if (this.onChunk) this.onChunk(chunk);
            });
            
            this.errorCallbacks.set(messageId, (error) => {
                reject(error);
            });
            
            this.completeCallbacks.set(messageId, (data) => {
                resolve(data);
            });
            
            // Send the message
            this.socket.send(JSON.stringify({
                type: 'message',
                message_id: messageId,
                session_id: sessionId,
                content: message,
                model_id: modelId
            }));
        });
    }

    // Set a callback to handle streaming chunks
    onStreamingChunk(callback) {
        this.onChunk = callback;
    }

    disconnect() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }
} 