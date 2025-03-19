import { html, css, LitElement } from 'lit';
import { repeat } from 'lit/directives/repeat.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import ChatAPI from '../api/chat-api.js';
import FilesAPI from '../api/files-api.js';
import { StreamingResponseHandler } from './streaming-response-handler.js';
import './file-details-viewer.js';
import './file-preview.js';
import './evidence-explorer.js';
import './mention-provider.js';

/**
 * Chat Interface Component
 * 
 * Vereinfachte Chat-Komponente ohne API-Testing Funktionalitäten
 */
class ChatInterface extends LitElement {
    static properties = {
        sessions: { type: Array },
        messages: { type: Array },
        activeSessionId: { type: String },
        messageInput: { type: String },
        loading: { type: Boolean },
        sendingMessage: { type: Boolean },
        error: { type: String },
        attachments: { type: Array },
        focusMode: { type: Boolean },
        suggestions: { type: Array },
        messageTypes: { type: Object },
        rewriteMessageId: { type: String },
        loadingMessages: { type: Array },
        typingIndicator: { type: Boolean },
        editingMessageId: { type: String },
        pinnedMessages: { type: Array },
        threadView: { type: Boolean },
        selectedThread: { type: String },
        chatTitle: { type: String },
        focusModeType: { type: String }, // 'none', 'minimal', 'zen', 'full'
        sharedMessages: { type: Array },
        copilotMode: { type: Boolean },
        copilotSuggestion: { type: String },
        copilotEnabled: { type: Boolean },
        performanceMetrics: { type: Object },
        debugInfo: { type: Object },
        sessionId: { type: String },
        streamingResponse: { type: String },
        selectedFile: { type: Object },
        availableModels: { type: Array },
        selectedModel: { type: String },
        contextFiles: { type: Array },
        additionalContext: { type: Array }
    };
    
    constructor() {
        super();
        this.sessions = [];
        this.messages = [];
        this.activeSessionId = null;
        this.messageInput = '';
        this.loading = false;
        this.sendingMessage = false;
        this.error = null;
        this.attachments = [];
        this.focusMode = false;
        this.suggestions = [];
        this.messageTypes = {
            SYSTEM: 'system',
            USER: 'user',
            ASSISTANT: 'assistant'
        };
        this.rewriteMessageId = null;
        this.loadingMessages = [];
        this.typingIndicator = false;
        this.editingMessageId = null;
        this.pinnedMessages = [];
        this.threadView = false;
        this.selectedThread = null;
        this.chatTitle = '';
        this.focusModeType = 'none';
        this.sharedMessages = [];
        this.copilotMode = false;
        this.copilotSuggestion = '';
        this.copilotEnabled = false;
        this.performanceMetrics = {
            avgResponseTime: 0,
            messageCount: 0,
            tokenUsage: 0,
            lastUpdateTime: null,
            cacheHitRate: 0,
            dbQueryCount: 0,
            memoryUsage: 0,
            activeConnections: 0,
            wsLatency: 0,
            apiCallsPerMinute: 0,
            errorRate: 0
        };
        this.debugInfo = {
            lastError: null,
            apiStatus: 'ok',
            wsStatus: 'connected',
            modelInfo: null,
            dbStatus: {
                connections: 0,
                queryLog: [],
                slowQueries: [],
                indexUsage: {}
            },
            cacheStatus: {
                size: 0,
                hits: 0,
                misses: 0,
                keys: []
            },
            systemStatus: {
                cpu: 0,
                memory: 0,
                uptime: 0
            }
        };
        this.sessionId = '';
        this.streamingResponse = '';
        this.selectedFile = null;
        this.availableModels = [];
        this.selectedModel = '';
        this.contextFiles = [];
        this.additionalContext = [];
        
        this.streamingHandler = new StreamingResponseHandler();
        this.streamingHandler.onStreamingChunk(this.handleStreamingChunk.bind(this));
        
        // Mention Provider Referenz
        this.mentionProvider = null;
    }

    firstUpdated() {
        this.loadSessions();
        this.setupImageUpload();  // Initialize image upload
        
        // Initialize WebSocket
        this.connectToWebSocket();
        
        // Enable keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+P or Cmd+P to toggle performance metrics
            if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
                e.preventDefault();
                this.toggleDebugPanel();
            }
        });
    }

    // Lifecycle methods
    connectedCallback() {
        super.connectedCallback();
        this.loadSessions();
        this.loadAvailableModels();
        this.setupWebSocket();
        
        if (this.ws) {
            this.ws.addEventListener('message', this.handleTypingStatus.bind(this));
        }
    }

    updated(changedProperties) {
        if (changedProperties.has('messages')) {
            this.scrollToBottom();
        }
    }

    // Core chat functionality
    async loadSessions() {
        try {
            const sessions = await ChatAPI.getSessions();
            this.sessions = sessions;
            if (sessions.length > 0 && !this.activeSessionId) {
                this.selectSession(sessions[0].id);
            }
        } catch (error) {
            this.error = 'Failed to load chat sessions';
            console.error(error);
        }
    }

    async selectSession(sessionId) {
        if (this.activeSessionId === sessionId) return;
        
        try {
            this.activeSessionId = sessionId;
            this.messages = [];
            const messages = await ChatAPI.getMessages(sessionId);
            this.messages = messages;
            this.scrollToBottom();
        } catch (error) {
            this.error = 'Failed to load chat messages';
            console.error(error);
        }
    }

    // Focus Mode
    toggleFocusMode() {
        this.focusMode = !this.focusMode;
        document.body.classList.toggle('focus-mode', this.focusMode);
    }

    // Message Rewrite
    async rewriteMessage(messageId) {
        this.rewriteMessageId = messageId;
        const message = this.messages.find(m => m.id === messageId);
        if (!message) return;

        try {
            const rewrittenMessage = await ChatAPI.rewriteMessage(
                this.activeSessionId,
                messageId
            );
            
            // Replace old message with rewritten one
            this.messages = this.messages.map(m => 
                m.id === messageId ? rewrittenMessage : m
            );
        } catch (error) {
            this.error = 'Failed to rewrite message';
            console.error(error);
        } finally {
            this.rewriteMessageId = null;
        }
    }

    // Auto-Complete & Suggestions
    async updateSuggestions(input) {
        if (!input || input.length < 2) {
            this.suggestions = [];
            return;
        }

        try {
            this.suggestions = await ChatAPI.getSuggestions(input);
        } catch (error) {
            console.error('Failed to get suggestions:', error);
            this.suggestions = [];
        }
    }

    applySuggestion(suggestion) {
        this.messageInput = suggestion;
        this.suggestions = [];
    }

    // Loading States
    addLoadingMessage() {
        const loadingId = `loading-${Date.now()}`;
        this.loadingMessages = [...this.loadingMessages, loadingId];
        return loadingId;
    }

    removeLoadingMessage(loadingId) {
        this.loadingMessages = this.loadingMessages.filter(id => id !== loadingId);
    }

    // Enhanced Message Sending
    async sendMessage(event) {
        event?.preventDefault();
        
        const input = this.shadowRoot.querySelector('#message-input').value.trim();
        if (!input && this.attachedImages.length === 0) return;
        
        // Clear input
        this.shadowRoot.querySelector('#message-input').value = '';
        
        // Add loading message
        const loadingId = this.addLoadingMessage();
        
        // Create form data for file upload if we have images
        let uploadedImagePaths = [];
        if (this.attachedImages.length > 0) {
            try {
                const formData = new FormData();
                this.attachedImages.forEach(file => {
                    formData.append('images', file);
                });
                
                // Upload images first
                const uploadResponse = await fetch('/api/upload/', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                    }
                });
                
                if (!uploadResponse.ok) {
                    throw new Error('Failed to upload images');
                }
                
                const uploadData = await uploadResponse.json();
                uploadedImagePaths = uploadData.file_paths || [];
                
            } catch (error) {
                console.error('Error uploading images:', error);
                this.showNotification('Error uploading images: ' + error.message);
                this.removeLoadingMessage(loadingId);
                return;
            }
        }
        
        // Add user message to UI
        const messageObj = {
            id: Date.now().toString(),
            role: 'user',
            content: input,
            timestamp: new Date().toISOString(),
            sender: {
                id: 'user',
                name: 'You'
            },
            images: uploadedImagePaths
        };
        
        // Store message locally
        if (!this.messages) this.messages = [];
        this.messages.push(messageObj);
        
        // Determine if we need to use multimodal endpoint
        const isMultimodal = uploadedImagePaths.length > 0;
        let apiEndpoint = '/api/chat/';
        let requestBody = {
            message: input,
            session_id: this.sessionId
        };
        
        if (isMultimodal) {
            apiEndpoint = '/api/chat/multimodal/';
            requestBody = {
                message: input,
                images: uploadedImagePaths,
                session_id: this.sessionId,
                model_choice: 'qwen'  // Can be made configurable
            };
        }
        
        try {
            // Send message to server
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Remove loading message
            this.removeLoadingMessage(loadingId);
            
            // Add AI response to UI
            const aiResponse = {
                id: Date.now().toString(),
                role: 'assistant',
                content: data.response || 'No response from server',
                timestamp: new Date().toISOString(),
                sender: {
                    id: 'ai',
                    name: 'AI'
                }
            };
            
            // Add AI message locally
            this.messages.push(aiResponse);
            
            // Process the message (adds to DOM)
            this.processMessage(aiResponse);
            
            // Clear attached images
            this.attachedImages = [];
            this.imagePreviewContainer.innerHTML = '';
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.removeLoadingMessage(loadingId);
            this.showNotification('Error: ' + error.message);
        }
    }

    // UI helpers
    scrollToBottom() {
        const chatMessages = this.shadowRoot.querySelector('.chat-messages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    renderMarkdown(content) {
        const html = marked.parse(content);
        const sanitized = DOMPurify.sanitize(html);
        return unsafeHTML(sanitized);
    }

    // Event handlers
    handleInputChange(event) {
        this.messageInput = event.target.value;
        
        // Überprüfen, ob ein @-Zeichen eingegeben wurde
        const text = event.target.value;
        const cursorPos = event.target.selectionStart;
        const lastChar = text.charAt(cursorPos - 1);
        
        if (lastChar === '@') {
            // Position für das Dropdown-Menü berechnen
            const textArea = event.target;
            const caretPos = this.getCaretCoordinates(textArea, textArea.selectionStart);
            const textAreaRect = textArea.getBoundingClientRect();
            
            const position = {
                top: textAreaRect.top + caretPos.top + 20, // 20px unterhalb des Cursors
                left: textAreaRect.left + caretPos.left
            };
            
            // Mention Provider anzeigen
            this.mentionProvider.show(textArea, position);
        }
    }

    handleInputKeyDown(event) {
        // Weiterleiten an MentionProvider, wenn es geöffnet ist
        if (this.mentionProvider && this.mentionProvider.isOpen) {
            this.mentionProvider.handleKeyDown(event);
        }
    }

    // Hilfsfunktion, um die Cursor-Position im Textfeld zu bestimmen
    getCaretCoordinates(element, position) {
        // Erstellen eines temporären Span-Elements zur Positionsberechnung
        const div = document.createElement('div');
        const span = document.createElement('span');
        const text = element.value.substring(0, position);
        
        // Stil-Eigenschaften vom Original-Element kopieren
        const computedStyle = window.getComputedStyle(element);
        const properties = [
            'fontFamily', 'fontSize', 'fontWeight', 'letterSpacing',
            'lineHeight', 'textIndent', 'wordSpacing', 'paddingLeft', 'paddingTop',
            'paddingRight', 'paddingBottom', 'width'
        ];
        
        properties.forEach(prop => {
            div.style[prop] = computedStyle[prop];
        });
        
        // Positionieren des temporären Divs
        div.style.position = 'absolute';
        div.style.visibility = 'hidden';
        div.style.whiteSpace = 'pre-wrap';
        div.textContent = text;
        span.textContent = '.';
        div.appendChild(span);
        document.body.appendChild(div);
        
        // Position berechnen
        const rect = span.getBoundingClientRect();
        const coordinates = {
            top: rect.top - div.getBoundingClientRect().top,
            left: rect.left - div.getBoundingClientRect().left
        };
        
        // Aufräumen
        document.body.removeChild(div);
        
        return coordinates;
    }

    // Event-Handler für die Auswahl einer Mention
    handleMentionSelected(event) {
        const mention = event.detail.mention;
        
        // Kontext für den Chat erweitern
        this.addMentionContext(mention);
    }

    // Kontext für den Chat um die Mention erweitern
    async addMentionContext(mention) {
        // Daten für das ausgewählte Element abrufen
        try {
            const response = await fetch(`/models/api/mentions/${mention.type}/${mention.id}`);
            if (response.ok) {
                const itemDetails = await response.json();
                
                // Kontext hinzufügen (wird beim nächsten Senden mitgeschickt)
                if (!this.additionalContext) {
                    this.additionalContext = [];
                }
                
                this.additionalContext.push({
                    type: 'mention',
                    category: mention.type,
                    id: mention.id,
                    name: mention.text,
                    data: itemDetails
                });
                
                // Hinweis anzeigen, dass der Kontext erweitert wurde
                this.showNotification(`Kontext erweitert mit: ${mention.text}`);
            }
        } catch (error) {
            console.error('Fehler beim Abrufen der Mention-Details:', error);
        }
    }

    // Hinweis anzeigen
    showNotification(message) {
        // Implementierung eines temporären Hinweises
        const notification = document.createElement('div');
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: var(--primary-color, #0366d6);
            color: white;
            padding: 10px 15px;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
            transition: opacity 0.3s;
        `;
        
        document.body.appendChild(notification);
        
        // Nach 3 Sekunden ausblenden und entfernen
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    // Message Threading
    async startThread(messageId) {
        this.selectedThread = messageId;
        this.threadView = true;
        try {
            const threadMessages = await ChatAPI.getThreadMessages(messageId);
            this.messages = threadMessages;
        } catch (error) {
            this.error = 'Failed to load thread';
            console.error(error);
        }
    }

    // Message Editing
    async editMessage(messageId) {
        this.editingMessageId = messageId;
        const message = this.messages.find(m => m.id === messageId);
        if (!message) return;
        this.messageInput = message.content;
    }

    async saveEdit() {
        if (!this.editingMessageId) return;

        try {
            const updatedMessage = await ChatAPI.updateMessage(
                this.editingMessageId,
                this.messageInput
            );
            
            this.messages = this.messages.map(m => 
                m.id === this.editingMessageId ? updatedMessage : m
            );
            this.messageInput = '';
            this.editingMessageId = null;
        } catch (error) {
            this.error = 'Failed to update message';
            console.error(error);
        }
    }

    // Message Pinning
    togglePinMessage(messageId) {
        const isPinned = this.pinnedMessages.includes(messageId);
        if (isPinned) {
            this.pinnedMessages = this.pinnedMessages.filter(id => id !== messageId);
        } else {
            this.pinnedMessages = [...this.pinnedMessages, messageId];
        }
    }

    // Typing Indicator
    startTyping() {
        if (!this.typingIndicator) {
            this.typingIndicator = true;
            ChatAPI.sendTypingStatus(this.activeSessionId, true);
        }
    }

    stopTyping() {
        if (this.typingIndicator) {
            this.typingIndicator = false;
            ChatAPI.sendTypingStatus(this.activeSessionId, false);
        }
    }

    // Enhanced Message Rendering
    renderMessage(message) {
        const isEditing = this.editingMessageId === message.id;
        const isPinned = this.pinnedMessages.includes(message.id);
        const isStreaming = message.streaming && message.role === 'assistant';

        return html`
            <div class="message ${message.role} ${isEditing ? 'editing' : ''} ${isPinned ? 'pinned' : ''}">
                ${this.renderMessageHeader(message)}
                
                <div class="message-content">
                    ${isEditing ? html`
                        <textarea
                            .value="${this.messageInput}"
                            @input="${this.handleInputChange}"
                            @keypress="${this.handleKeyPress}"
                        ></textarea>
                        <div class="edit-actions">
                            <button @click="${this.saveEdit}">Save</button>
                            <button @click="${() => this.editingMessageId = null}">Cancel</button>
                        </div>
                    ` : html`
                        ${isStreaming 
                            ? html`<div class="streaming-content">${this.streamingResponse}</div>` 
                            : html`${this.renderMarkdown(message.content)}`
                        }
                    `}
                </div>

                <div class="message-actions">
                    ${message.role === this.messageTypes.USER ? html`
                        <button @click="${() => this.editMessage(message.id)}">
                            Edit
                        </button>
                    ` : ''}
                    <button @click="${() => this.togglePinMessage(message.id)}">
                        ${isPinned ? 'Unpin' : 'Pin'}
                    </button>
                    <button @click="${() => this.startThread(message.id)}">
                        Thread
                    </button>
                </div>
            </div>
        `;
    }

    renderMessageHeader(message) {
        return html`
            <div class="message-header">
                <span class="message-timestamp">
                    ${this.formatTimestamp(message.timestamp)}
                </span>
                ${message.edited ? html`
                    <span class="message-edited">(edited)</span>
                ` : ''}
                ${this.pinnedMessages.includes(message.id) ? html`
                    <span class="message-pinned">📌</span>
                ` : ''}
            </div>
        `;
    }

    // Main render method
    render() {
        return html`
            <div class="chat-container ${this.focusMode ? 'focus-mode' : ''}">
                <div class="chat-sidebar">
                    <button @click="${this.createNewSession}" class="new-chat-btn">
                        New Chat
                    </button>
                    <div class="sessions-list">
                        ${this.sessions.map(session => html`
                            <div class="session-item ${session.id === this.activeSessionId ? 'active' : ''}"
                                 @click="${() => this.selectSession(session.id)}">
                                ${session.title}
                            </div>
                        `)}
                    </div>
                </div>
                
                <div class="chat-main">
                    ${this.threadView ? this.renderThreadView() : this.renderMainView()}
                </div>

                ${this.renderPinnedMessages()}
                
                ${this.renderContextFiles()}
            </div>
            
            ${this.error ? html`
                <div class="error-toast">
                    ${this.error}
                    <button @click="${() => this.error = null}">×</button>
                </div>
            ` : ''}
            
            <!-- Mention Provider hinzufügen -->
            <mention-provider 
                @mention-selected="${this.handleMentionSelected}"
            ></mention-provider>
        `;
    }

    static styles = css`
        .focus-mode {
            background: var(--bruno-focus-bg, #1a1a1a);
            color: var(--bruno-focus-text, #ffffff);
        }

        .loading-indicator {
            display: flex;
            gap: 4px;
            padding: 8px;
        }

        .dot {
            width: 8px;
            height: 8px;
            background: currentColor;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .suggestions {
            position: absolute;
            bottom: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid var(--bruno-border-color);
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }

        .suggestion {
            padding: 8px;
            cursor: pointer;
        }

        .suggestion:hover {
            background: var(--bruno-hover-bg);
        }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .message.pinned {
            border-left: 3px solid var(--bruno-accent-color, #ffd700);
        }

        .message.editing {
            background: var(--bruno-editing-bg, #f0f0f0);
        }

        .message-header {
            display: flex;
            gap: 8px;
            font-size: 0.8em;
            color: var(--bruno-text-secondary);
        }

        .edit-actions {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }

        .pinned-messages {
            position: fixed;
            top: 0;
            right: 0;
            max-width: 300px;
            background: white;
            border-left: 1px solid var(--bruno-border-color);
            padding: 16px;
        }

        .focus-mode-minimal {
            --bruno-bg: #fafafa;
            --bruno-text: #333;
        }

        .focus-mode-zen {
            --bruno-bg: #000;
            --bruno-text: #fff;
        }

        .focus-mode-full {
            --bruno-bg: #1a1a1a;
            --bruno-text: #fff;
            --bruno-sidebar-width: 0px;
        }

        .message-actions {
            opacity: 0;
            transition: opacity 0.2s;
        }

        .message:hover .message-actions {
            opacity: 1;
        }

        .copilot-container {
            position: fixed;
            right: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 300px;
            background: var(--bruno-bg-secondary);
            border-left: 1px solid var(--bruno-border-color);
            padding: 16px;
        }

        .copilot-suggestion {
            font-family: monospace;
            padding: 8px;
            background: var(--bruno-code-bg);
            border-radius: 4px;
            margin-top: 8px;
        }

        .debug-panel {
            position: fixed;
            bottom: 0;
            right: 0;
            background: var(--bruno-bg-secondary);
            padding: 16px;
            border-top-left-radius: 8px;
            border: 1px solid var(--bruno-border-color);
            max-width: 400px;
            max-height: 600px;
            overflow-y: auto;
        }

        .metrics-section, .debug-section {
            margin-bottom: 16px;
            padding: 8px;
            background: var(--bruno-bg-tertiary);
            border-radius: 4px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }

        .query-item {
            font-family: monospace;
            font-size: 0.8em;
            padding: 4px;
            background: var(--bruno-code-bg);
            margin: 2px 0;
            border-radius: 2px;
        }

        .slow-queries {
            max-height: 150px;
            overflow-y: auto;
        }

        .context-files {
            margin-top: 16px;
            border-top: 1px solid var(--border-color, #ddd);
            padding-top: 16px;
        }
        
        .context-files h3 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 16px;
            color: var(--heading-color, #333);
        }

        .image-upload-btn {
            background: transparent;
            border: none;
            cursor: pointer;
            padding: 6px;
            border-radius: 4px;
            color: var(--chat-icon-color, #666);
            margin-right: 6px;
        }
        
        .image-upload-btn:hover {
            background: var(--chat-icon-hover-bg, rgba(0,0,0,0.05));
        }
        
        .image-preview-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 8px;
            padding: 0 16px;
        }
        
        .image-preview {
            position: relative;
            width: 80px;
            height: 80px;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid var(--chat-border-color, #ddd);
        }
        
        .image-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .remove-image {
            position: absolute;
            top: 2px;
            right: 2px;
            background: rgba(0,0,0,0.5);
            color: white;
            border: none;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            font-size: 14px;
            line-height: 1;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .message-image {
            max-width: 300px;
            max-height: 200px;
            border-radius: 4px;
            margin: 4px 0;
            cursor: pointer;
        }
        
        @media (max-width: 768px) {
            .message-image {
                max-width: 200px;
                max-height: 150px;
            }
        }
    `;

    // Enhanced Focus Mode
    setFocusMode(type) {
        this.focusModeType = type;
        document.body.className = `focus-mode-${type}`;
    }

    // Share functionality
    async shareMessage(messageId) {
        const message = this.messages.find(m => m.id === messageId);
        if (!message) return;

        try {
            const shareUrl = await ChatAPI.createShareLink(messageId);
            await navigator.clipboard.writeText(shareUrl);
            this.sharedMessages = [...this.sharedMessages, messageId];
        } catch (error) {
            this.error = 'Failed to share message';
            console.error(error);
        }
    }

    // Auto Title Generation
    updateChatTitle() {
        if (this.messages.length > 0) {
            const firstMessage = this.messages[0].content;
            this.chatTitle = firstMessage.substring(0, 30) + '...';
            document.title = `${this.chatTitle} - Chat`;
        }
    }

    // Chat Export/Import
    exportChat() {
        const chatData = {
            title: this.chatTitle,
            messages: this.messages,
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(chatData)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-${this.activeSessionId}.json`;
        a.click();
    }

    async importChat(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            const text = await file.text();
            const chatData = JSON.parse(text);
            
            // Create new session with imported data
            const sessionId = await ChatAPI.createSession(chatData);
            this.selectSession(sessionId);
        } catch (error) {
            this.error = 'Failed to import chat';
            console.error(error);
        }
    }

    // Enhanced Message Actions
    renderMessageActions(message) {
        return html`
            <div class="message-actions">
                <button @click="${() => this.shareMessage(message.id)}">
                    ${this.sharedMessages.includes(message.id) ? 'Shared' : 'Share'}
                </button>
                <button @click="${() => this.editMessage(message.id)}">
                    Edit
                </button>
                <button @click="${() => this.togglePinMessage(message.id)}">
                    ${this.pinnedMessages.includes(message.id) ? 'Unpin' : 'Pin'}
                </button>
                <button @click="${() => this.startThread(message.id)}">
                    Thread
                </button>
            </div>
        `;
    }

    // Focus Mode Selector
    renderFocusModeSelector() {
        return html`
            <div class="focus-mode-selector">
                <select @change="${(e) => this.setFocusMode(e.target.value)}">
                    <option value="none">Normal Mode</option>
                    <option value="minimal">Minimal</option>
                    <option value="zen">Zen Mode</option>
                    <option value="full">Full Focus</option>
                </select>
            </div>
        `;
    }

    // Copilot Mode
    toggleCopilot() {
        this.copilotMode = !this.copilotMode;
        if (this.copilotMode) {
            this.startCopilot();
        } else {
            this.stopCopilot();
        }
    }

    async startCopilot() {
        this.copilotEnabled = true;
        await ChatAPI.enableCopilot(this.activeSessionId);
    }

    stopCopilot() {
        this.copilotEnabled = false;
        this.copilotSuggestion = '';
        ChatAPI.disableCopilot(this.activeSessionId);
    }

    async updateCopilotSuggestion(input) {
        if (!this.copilotEnabled || input.length < 2) return;

        try {
            const suggestion = await ChatAPI.getCopilotSuggestion(input);
            this.copilotSuggestion = suggestion;
        } catch (error) {
            console.error('Copilot suggestion failed:', error);
            this.copilotSuggestion = '';
        }
    }

    // Performance Monitoring
    updatePerformanceMetrics(data) {
        const now = Date.now();
        this.performanceMetrics = {
            ...this.performanceMetrics,
            messageCount: this.performanceMetrics.messageCount + 1,
            avgResponseTime: this.calculateAvgResponseTime(data.timestamp),
            tokenUsage: this.calculateTokenUsage(data.content),
            lastUpdateTime: now,
            cacheHitRate: data.cacheStats?.hitRate || 0,
            dbQueryCount: data.dbStats?.queryCount || 0,
            memoryUsage: data.systemStats?.memoryUsage || 0,
            activeConnections: data.networkStats?.connections || 0,
            wsLatency: data.networkStats?.wsLatency || 0,
            apiCallsPerMinute: data.apiStats?.callsPerMinute || 0,
            errorRate: data.errorStats?.rate || 0
        };
    }

    // Debug Information
    updateDebugInfo(info) {
        this.debugInfo = {
            ...this.debugInfo,
            ...info,
            lastUpdate: new Date().toISOString(),
            dbStatus: {
                connections: info.dbStats?.connections || 0,
                queryLog: info.dbStats?.recentQueries || [],
                slowQueries: info.dbStats?.slowQueries || [],
                indexUsage: info.dbStats?.indexStats || {}
            },
            cacheStatus: {
                size: info.cacheStats?.size || 0,
                hits: info.cacheStats?.hits || 0,
                misses: info.cacheStats?.misses || 0,
                keys: info.cacheStats?.keys || []
            },
            systemStatus: {
                cpu: info.systemStats?.cpuUsage || 0,
                memory: info.systemStats?.memoryUsage || 0,
                uptime: info.systemStats?.uptime || 0
            }
        };
    }

    renderCopilotView() {
        if (!this.copilotMode) return '';

        return html`
            <div class="copilot-container ${this.copilotEnabled ? 'active' : ''}">
                <div class="copilot-header">
                    <span>Copilot Mode ${this.copilotEnabled ? '(Active)' : '(Inactive)'}</span>
                    <button @click="${this.toggleCopilot}">
                        ${this.copilotEnabled ? 'Disable' : 'Enable'} Copilot
                    </button>
                </div>
                <div class="copilot-suggestion">
                    ${this.copilotSuggestion}
                </div>
            </div>
        `;
    }

    renderDebugPanel() {
        return html`
            <div class="debug-panel">
                <h3>Debug Information</h3>
                
                <div class="metrics-section">
                    <h4>Performance</h4>
                    <div class="metrics-grid">
                        <div>Messages: ${this.performanceMetrics.messageCount}</div>
                        <div>Avg Response: ${this.performanceMetrics.avgResponseTime}ms</div>
                        <div>Cache Hit Rate: ${this.performanceMetrics.cacheHitRate}%</div>
                        <div>DB Queries: ${this.performanceMetrics.dbQueryCount}</div>
                        <div>Memory: ${this.formatBytes(this.performanceMetrics.memoryUsage)}</div>
                        <div>WS Latency: ${this.performanceMetrics.wsLatency}ms</div>
                        <div>API Calls/min: ${this.performanceMetrics.apiCallsPerMinute}</div>
                        <div>Error Rate: ${this.performanceMetrics.errorRate}%</div>
                    </div>
                </div>

                <div class="debug-section">
                    <h4>Database</h4>
                    <div>Connections: ${this.debugInfo.dbStatus.connections}</div>
                    <div class="slow-queries">
                        <h5>Slow Queries</h5>
                        ${this.debugInfo.dbStatus.slowQueries.map(query => html`
                            <div class="query-item">${query.sql} (${query.duration}ms)</div>
                        `)}
                    </div>
                </div>

                <div class="debug-section">
                    <h4>Cache</h4>
                    <div>Size: ${this.formatBytes(this.debugInfo.cacheStatus.size)}</div>
                    <div>Hit Ratio: ${this.calculateHitRatio()}%</div>
                </div>

                <div class="debug-section">
                    <h4>System</h4>
                    <div>CPU: ${this.debugInfo.systemStatus.cpu}%</div>
                    <div>Memory: ${this.formatBytes(this.debugInfo.systemStatus.memory)}</div>
                    <div>Uptime: ${this.formatUptime(this.debugInfo.systemStatus.uptime)}</div>
                </div>
            </div>
        `;
    }

    // Utility Funktionen
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
    }

    formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${days}d ${hours}h ${minutes}m`;
    }

    calculateHitRatio() {
        const { hits, misses } = this.debugInfo.cacheStatus;
        const total = hits + misses;
        return total === 0 ? 0 : ((hits / total) * 100).toFixed(1);
    }

    async loadAvailableModels() {
        try {
            const response = await fetch('/api/models/');
            if (response.ok) {
                const data = await response.json();
                this.availableModels = data.models;
                this.selectedModel = data.default_model || (this.availableModels[0]?.id || '');
            } else {
                console.error('Failed to load models:', await response.text());
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    handleStreamingChunk(chunk) {
        this.streamingResponse += chunk;
        // Scroll to bottom to show new content
        const chatContainer = this.shadowRoot.querySelector('.chat-messages');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    selectFile(file) {
        this.selectedFile = file;
    }

    // Add a method to add context files
    addContextFile(filePath) {
        // Check if file is already in context
        if (!this.contextFiles.some(file => file.path === filePath)) {
            this.contextFiles = [
                ...this.contextFiles, 
                { path: filePath, name: filePath.split('/').pop() }
            ];
        }
    }
    
    // Add a method to remove context files
    removeContextFile(filePath) {
        this.contextFiles = this.contextFiles.filter(file => file.path !== filePath);
    }
    
    // Render context files
    renderContextFiles() {
        if (this.contextFiles.length === 0) {
            return '';
        }
        
        return html`
            <div class="context-files">
                <h3>Context Files</h3>
                ${this.contextFiles.map(file => html`
                    <file-preview 
                        filePath=${file.path}
                        fileName=${file.name}
                    ></file-preview>
                `)}
            </div>
        `;
    }

    renderMessageContent(message) {
        if (message.is_loading) {
            return html`<div class="loading-dots"><span></span><span></span><span></span></div>`;
        }
        
        // Render message images if any
        const imageElements = message.images && message.images.length > 0 
            ? message.images.map(img => html`
                <img class="message-image" src="${img}" alt="Attached image" 
                    @click=${() => this.openImagePreview(img)}>
            `) 
            : '';
        
        return html`
            ${imageElements}
            ${this.renderMarkdown(message.content)}
        `;
    }

    // Method to open full-size image preview
    openImagePreview(imageSrc) {
        const modal = document.createElement('div');
        modal.className = 'image-preview-modal';
        modal.style = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;
        
        const img = document.createElement('img');
        img.src = imageSrc;
        img.style = `
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        `;
        
        modal.appendChild(img);
        document.body.appendChild(modal);
        
        // Close on click
        modal.addEventListener('click', () => {
            modal.remove();
        });
    }

    // Handle typing status messages from WebSocket
    handleTypingStatus(event) {
        const data = JSON.parse(event.data);
        
        if (data.type === 'typing_status') {
            const { is_typing, user_id } = data;
            
            // Update UI to show/hide typing indicator
            if (is_typing) {
                this.showTypingIndicator(user_id);
            } else {
                this.hideTypingIndicator(user_id);
            }
        }
    }

    // Show typing indicator in the UI
    showTypingIndicator(userId) {
        // Find or create typing indicator for this user
        let indicator = this.shadowRoot.querySelector(`.typing-indicator[data-user="${userId}"]`);
        
        if (!indicator) {
            const chatContainer = this.shadowRoot.querySelector('.chat-messages');
            indicator = document.createElement('div');
            indicator.className = 'typing-indicator';
            indicator.setAttribute('data-user', userId);
            indicator.innerHTML = '<span></span><span></span><span></span>';
            chatContainer.appendChild(indicator);
        }
        
        // Show the indicator
        indicator.style.display = 'block';
    }

    // Hide typing indicator in the UI
    hideTypingIndicator(userId) {
        const indicator = this.shadowRoot.querySelector(`.typing-indicator[data-user="${userId}"]`);
        if (indicator) {
            indicator.style.display = 'none';
        }
    }

    // Handle user input and set up typing status
    handleUserInput(event) {
        // Existing code for handling input
        
        // Send typing status
        if (!this.typingTimeout) {
            this.sendTypingStatus(true);
        }
        
        // Clear existing timeout and set a new one
        clearTimeout(this.typingTimeout);
        this.typingTimeout = setTimeout(() => {
            this.sendTypingStatus(false);
            this.typingTimeout = null;
        }, 2000); // Stop typing indicator after 2 seconds of inactivity
    }

    // Send typing status to server
    sendTypingStatus(isTyping) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'typing',
                is_typing: isTyping
            }));
        }
    }

    async handleAIResponse(message) {
        // Get the query_id from the message
        const queryId = message.response.query_id;
        
        if (queryId) {
            // Update evidence explorer with the new query ID
            const evidenceExplorer = this.shadowRoot.querySelector('evidence-explorer');
            if (evidenceExplorer) {
                evidenceExplorer.queryId = queryId;
            } else {
                // Create and append evidence explorer if it doesn't exist
                const explorer = document.createElement('evidence-explorer');
                explorer.queryId = queryId;
                
                // Find the appropriate message container and append the explorer
                const messageContainer = this.shadowRoot.querySelector(`.message[data-id="${message.id}"]`);
                if (messageContainer) {
                    messageContainer.appendChild(explorer);
                }
            }
        }
        
        // Existing response handling code...
    }

    // Optional: Methode zum manuellen Hinzufügen der Evidence Explorer
    addEvidenceExplorer(messageId, queryId) {
        const messageElement = this.shadowRoot.querySelector(`.message[data-id="${messageId}"]`);
        if (!messageElement) return;
        
        const existingExplorer = messageElement.querySelector('evidence-explorer');
        if (existingExplorer) {
            existingExplorer.queryId = queryId;
            return;
        }
        
        const messageContent = messageElement.querySelector('.ai-message');
        if (!messageContent) return;
        
        const explorer = document.createElement('evidence-explorer');
        explorer.queryId = queryId;
        explorer.responseContent = messageContent.textContent.trim();
        
        messageContent.appendChild(explorer);
    }

    async processMessage(message) {
        // Existing code...
        
        // Check for @Web mentions that should trigger automatic web search
        const webMentionRegex = /@Web\b/i;
        if (webMentionRegex.test(message)) {
            // Extract search query - the entire message minus the @Web mention
            const searchQuery = message.replace(webMentionRegex, '').trim();
            
            try {
                // Send the query for automatic web search
                const response = await fetch('/api/mentions/web/auto-search/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: searchQuery,
                        category: 'web', // Default to web search
                        message_id: this.currentMessageId
                    }),
                });
                
                if (!response.ok) {
                    console.error('Error with auto web search:', await response.text());
                }
                
                // Web search is handled asynchronously - results will be added to context
                // We don't need to update the UI here
            } catch (error) {
                console.error('Failed to perform auto web search:', error);
            }
        }
        
        // Continue with regular message processing...
    }

    renderActions() {
        return html`
            <div class="chat-actions">
                <button @click="${this.sendMessage}" ?disabled="${!this.messageInput.trim() || this.loading}">
                    ${this.loading ? 'Sending...' : 'Send'}
                </button>
                <button @click="${this.startDeepResearch}" ?disabled="${!this.messageInput.trim() || this.loading}">
                    Deep Research
                </button>
            </div>
        `;
    }

    async startDeepResearch() {
        const query = this.messageInput.trim();
        if (!query) return;
        
        // Create loading state
        const loadingId = this.addLoadingMessage("Starting Deep Research...");
        
        try {
            // Call Deep Research API
            const response = await fetch('/api/search/deep-research/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this._getCsrfToken()
                },
                body: JSON.stringify({
                    query: query,
                    max_iterations: 3,
                    depth: 2,
                    breadth: 2
                })
            });
            
            if (!response.ok) {
                throw new Error("Failed to start Deep Research");
            }
            
            const data = await response.json();
            const researchId = data.research_id;
            
            // Poll for progress
            await this.pollDeepResearchProgress(researchId, loadingId);
        } catch (error) {
            console.error("Deep Research error:", error);
            this.removeLoadingMessage(loadingId);
            this.showNotification("Error starting Deep Research: " + error.message);
        }
    }

    async pollDeepResearchProgress(researchId, loadingId) {
        try {
            const response = await fetch(`/api/search/deep-research/${researchId}/status`);
            if (!response.ok) {
                throw new Error("Failed to check research progress");
            }
            
            const data = await response.json();
            
            // Update loading message
            const loadingMessage = this.shadowRoot.querySelector(`#${loadingId}`);
            if (loadingMessage) {
                loadingMessage.textContent = `Deep Research: ${data.status} (${data.progress}%)`;
            }
            
            if (data.status === 'completed') {
                // Get the results
                await this.getDeepResearchResults(researchId, loadingId);
            } else if (data.status === 'failed') {
                throw new Error("Deep Research failed");
            } else {
                // Continue polling
                setTimeout(() => this.pollDeepResearchProgress(researchId, loadingId), 2000);
            }
        } catch (error) {
            console.error("Error polling progress:", error);
            this.removeLoadingMessage(loadingId);
            this.showNotification("Error in Deep Research: " + error.message);
        }
    }

    async getDeepResearchResults(researchId, loadingId) {
        try {
            const response = await fetch(`/api/search/deep-research/${researchId}/results`);
            if (!response.ok) {
                throw new Error("Failed to get research results");
            }
            
            const data = await response.json();
            
            // Remove loading message
            this.removeLoadingMessage(loadingId);
            
            // Add the research findings as a special message
            this.messages = [...this.messages, {
                id: `deep-research-${Date.now()}`,
                content: data.findings[0].summary, // The main summary
                role: 'assistant',
                timestamp: new Date().toISOString(),
                type: 'deep-research',
                metadata: {
                    findings: data.findings,
                    original_query: this.messageInput
                }
            }];
            
            // Clear input
            this.messageInput = '';
            
            // Scroll to bottom
            this.scrollToBottom();
        } catch (error) {
            console.error("Error getting research results:", error);
            this.removeLoadingMessage(loadingId);
            this.showNotification("Error getting Deep Research results: " + error.message);
        }
    }

    // Add image upload functionality
    setupImageUpload() {
        // Create image upload button
        const imageUploadBtn = document.createElement('button');
        imageUploadBtn.className = 'image-upload-btn';
        imageUploadBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"></path><path d="M16 5h6v6"></path><path d="M8 12l3 3 6-6"></path></svg>';
        imageUploadBtn.title = 'Attach Image';
        
        // Create hidden file input
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.multiple = true;
        fileInput.style.display = 'none';
        fileInput.id = 'image-upload-input';
        
        // Add elements to DOM
        const inputContainer = this.shadowRoot.querySelector('.input-container');
        inputContainer.appendChild(imageUploadBtn);
        inputContainer.appendChild(fileInput);
        
        // Setup event listeners
        imageUploadBtn.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', this.handleImageUpload.bind(this));
        
        // Container for image previews
        this.imagePreviewContainer = document.createElement('div');
        this.imagePreviewContainer.className = 'image-preview-container';
        inputContainer.parentNode.insertBefore(this.imagePreviewContainer, inputContainer);
        
        // Initialize attachedImages array
        this.attachedImages = [];
    }
    
    // Handle image upload
    handleImageUpload(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;
        
        // Clear previous images if we're not in a thread
        if (!this.isInThread) {
            this.imagePreviewContainer.innerHTML = '';
            this.attachedImages = [];
        }
        
        // Process each file
        Array.from(files).forEach(file => {
            // Validate file type
            if (!file.type.match('image.*')) {
                this.showNotification('Only image files are supported.');
                return;
            }
            
            // Create a preview
            const reader = new FileReader();
            reader.onload = (e) => {
                const previewContainer = document.createElement('div');
                previewContainer.className = 'image-preview';
                
                const image = document.createElement('img');
                image.src = e.target.result;
                
                const removeBtn = document.createElement('button');
                removeBtn.className = 'remove-image';
                removeBtn.innerHTML = '×';
                removeBtn.addEventListener('click', () => {
                    // Remove this image
                    const index = this.attachedImages.findIndex(img => img.name === file.name);
                    if (index !== -1) {
                        this.attachedImages.splice(index, 1);
                    }
                    previewContainer.remove();
                });
                
                previewContainer.appendChild(image);
                previewContainer.appendChild(removeBtn);
                this.imagePreviewContainer.appendChild(previewContainer);
                
                // Store the file for upload
                this.attachedImages.push(file);
            };
            
            reader.readAsDataURL(file);
        });
        
        // Reset the input
        event.target.value = '';
    }
}

customElements.define('bruno-chat-interface', ChatInterface);

export default ChatInterface;