/**
 * Chat API Module
 * 
 * Provides functions for interacting with the chat API
 */
const ChatAPI = {
    /**
     * Get all chat sessions for the user
     * @returns {Promise} Promise that resolves to the list of chat sessions
     */
    getSessions: async function() {
        try {
            const response = await fetch('/api/chat/sessions/');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching chat sessions:', error);
            throw error;
        }
    },
    
    /**
     * Create a new chat session
     * @param {string} title - The title of the chat session
     * @param {string} model - The model to use for the chat session
     * @returns {Promise} Promise that resolves to the created chat session
     */
    createSession: async function(title, model) {
        try {
            const response = await fetch('/api/chat/sessions/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: title,
                    model: model
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error creating chat session:', error);
            throw error;
        }
    },
    
    /**
     * Get messages for a chat session
     * @param {string} sessionId - The ID of the chat session
     * @returns {Promise} Promise that resolves to the list of messages
     */
    getMessages: async function(sessionId) {
        try {
            const response = await fetch(`/api/chat/sessions/${sessionId}/messages/`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching messages:', error);
            throw error;
        }
    },
    
    /**
     * Send a message to a chat session
     * @param {string} sessionId - The ID of the chat session
     * @param {string} content - The content of the message
     * @param {Array} attachments - Optional array of file IDs to attach
     * @returns {Promise} Promise that resolves to the sent message and AI response
     */
    sendMessage: async function(sessionId, content, attachments = []) {
        try {
            const response = await fetch(`/api/chat/sessions/${sessionId}/messages/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: content,
                    attachments: attachments
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error sending message:', error);
            throw error;
        }
    },
    
    /**
     * Delete a chat session
     * @param {string} sessionId - The ID of the chat session to delete
     * @returns {Promise} Promise that resolves when the session is deleted
     */
    deleteSession: async function(sessionId) {
        try {
            const response = await fetch(`/api/chat/sessions/${sessionId}/`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return true;
        } catch (error) {
            console.error('Error deleting chat session:', error);
            throw error;
        }
    },
    
    /**
     * Update a chat session
     * @param {string} sessionId - The ID of the chat session to update
     * @param {Object} data - The data to update (title, system_message, etc.)
     * @returns {Promise} Promise that resolves to the updated chat session
     */
    updateSession: async function(sessionId, data) {
        try {
            const response = await fetch(`/api/chat/sessions/${sessionId}/`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error updating chat session:', error);
            throw error;
        }
    }
};

// Export the module
export default ChatAPI; 