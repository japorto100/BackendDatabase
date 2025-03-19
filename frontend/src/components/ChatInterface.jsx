import React, { useState, useEffect } from 'react';
import { Button, Input, Spin, Divider, Collapse } from 'antd';
import { SendOutlined, SettingOutlined } from '@ant-design/icons';
import axios from 'axios';
import ModelSelector from './ModelSelector';
import './ChatInterface.css';

const { TextArea } = Input;
const { Panel } = Collapse;

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [customModelUrl, setCustomModelUrl] = useState('');
  const [modelSettings, setModelSettings] = useState({
    temperature: 0.7,
    maxTokens: 1000
  });

  // Fetch chat history on component mount
  useEffect(() => {
    fetchChatHistory();
    fetchUserSettings();
  }, []);

  const fetchChatHistory = async () => {
    try {
      const response = await axios.get('/api/chat/history');
      setMessages(response.data.messages || []);
    } catch (err) {
      console.error('Error fetching chat history:', err);
    }
  };

  const fetchUserSettings = async () => {
    try {
      const response = await axios.get('/api/user/settings');
      const settings = response.data;
      
      setSelectedModel(settings.generation_model || '');
      setUseCustomModel(settings.use_custom_model || false);
      setCustomModelUrl(settings.custom_model_url || '');
      setModelSettings({
        temperature: settings.temperature || 0.7,
        maxTokens: settings.max_tokens || 1000
      });
    } catch (err) {
      console.error('Error fetching user settings:', err);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };
    
    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);
    
    try {
      const response = await axios.post('/api/chat/message', {
        message: input,
        model: selectedModel,
        use_custom_model: useCustomModel,
        custom_model_url: customModelUrl,
        temperature: modelSettings.temperature,
        max_tokens: modelSettings.maxTokens
      });
      
      const assistantMessage = {
        role: 'assistant',
        content: response.data.message,
        timestamp: new Date().toISOString(),
        citations: response.data.citations || []
      };
      
      setMessages(prevMessages => [...prevMessages, assistantMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
      
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
        error: true
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleModelChange = (model, useCustom, customUrl, settings) => {
    setSelectedModel(model);
    setUseCustomModel(useCustom);
    setCustomModelUrl(customUrl);
    setModelSettings(settings);
  };

  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>AI Chat</h2>
        <Button 
          icon={<SettingOutlined />} 
          onClick={toggleSettings}
          type={showSettings ? 'primary' : 'default'}
        >
          Model Settings
        </Button>
      </div>
      
      {showSettings && (
        <div className="chat-settings">
          <ModelSelector 
            onModelChange={handleModelChange}
            defaultModel={selectedModel}
            type="chat"
          />
        </div>
      )}
      
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <p>No messages yet. Start a conversation!</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div 
              key={index} 
              className={`message ${msg.role === 'user' ? 'user-message' : 'assistant-message'} ${msg.error ? 'error-message' : ''}`}
            >
              <div className="message-content">
                {msg.content}
              </div>
              
              {msg.citations && msg.citations.length > 0 && (
                <div className="message-citations">
                  <Collapse ghost>
                    <Panel header="Sources" key="1">
                      <ul>
                        {msg.citations.map((citation, idx) => (
                          <li key={idx}>
                            <a href={citation.url} target="_blank" rel="noopener noreferrer">
                              {citation.title}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </Panel>
                  </Collapse>
                </div>
              )}
              
              <div className="message-timestamp">
                {new Date(msg.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))
        )}
        
        {loading && (
          <div className="loading-message">
            <Spin size="small" />
            <span>AI is thinking...</span>
          </div>
        )}
      </div>
      
      <div className="chat-input">
        <TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          autoSize={{ minRows: 1, maxRows: 6 }}
          disabled={loading}
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={handleSend}
          disabled={!input.trim() || loading}
        >
          Send
        </Button>
      </div>
    </div>
  );
};

export default ChatInterface; 