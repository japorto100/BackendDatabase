import React, { useState, useEffect } from 'react';
import { Select, Checkbox, Input, Button, Tooltip, Divider, Alert, Spin } from 'antd';
import { InfoCircleOutlined, CloudDownloadOutlined, SettingOutlined } from '@ant-design/icons';
import axios from 'axios';
import './ModelSelector.css';

const { Option } = Select;

const ModelSelector = ({ onModelChange, defaultModel, type = 'chat' }) => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(defaultModel || '');
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [customModelUrl, setCustomModelUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [advancedSettings, setAdvancedSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1000);

  // Fetch available models on component mount
  useEffect(() => {
    fetchModels();
  }, [type]);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`/api/models/available?type=${type}`);
      setModels(response.data.models || []);
      
      // Set default model if none is selected
      if (!selectedModel && response.data.models && response.data.models.length > 0) {
        const defaultModel = response.data.default_model || response.data.models[0].id;
        setSelectedModel(defaultModel);
        if (onModelChange) onModelChange(defaultModel, false, '', { temperature, maxTokens });
      }
    } catch (err) {
      console.error('Error fetching models:', err);
      setError('Failed to load available models. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (value) => {
    setSelectedModel(value);
    if (onModelChange) {
      onModelChange(value, useCustomModel, customModelUrl, { temperature, maxTokens });
    }
  };

  const handleCustomModelToggle = (e) => {
    setUseCustomModel(e.target.checked);
    if (onModelChange) {
      onModelChange(selectedModel, e.target.checked, customModelUrl, { temperature, maxTokens });
    }
  };

  const handleCustomUrlChange = (e) => {
    setCustomModelUrl(e.target.value);
    if (useCustomModel && onModelChange) {
      onModelChange(selectedModel, useCustomModel, e.target.value, { temperature, maxTokens });
    }
  };

  const handleAdvancedSettingsToggle = () => {
    setAdvancedSettings(!advancedSettings);
  };

  const handleTemperatureChange = (value) => {
    setTemperature(value);
    if (onModelChange) {
      onModelChange(selectedModel, useCustomModel, customModelUrl, { temperature: value, maxTokens });
    }
  };

  const handleMaxTokensChange = (value) => {
    setMaxTokens(value);
    if (onModelChange) {
      onModelChange(selectedModel, useCustomModel, customModelUrl, { temperature, maxTokens: value });
    }
  };

  const saveSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      await axios.post('/api/user/settings', {
        [type === 'chat' ? 'generation_model' : 'hyde_model']: selectedModel,
        use_custom_model: useCustomModel,
        custom_model_url: customModelUrl,
        temperature,
        max_tokens: maxTokens
      });
      // Show success message or notification
    } catch (err) {
      console.error('Error saving settings:', err);
      setError('Failed to save settings. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="model-selector">
      {error && <Alert message={error} type="error" showIcon closable />}
      
      <div className="model-selector-main">
        <div className="model-select-container">
          <label>Select AI Model:</label>
          <Select
            value={selectedModel}
            onChange={handleModelChange}
            style={{ width: '100%' }}
            loading={loading}
            disabled={useCustomModel}
          >
            {models.map(model => (
              <Option key={model.id} value={model.id}>
                {model.name} 
                {model.provider && <span className="model-provider">({model.provider})</span>}
                {model.vision_capable && type === 'chat' && 
                  <Tooltip title="Supports image analysis">
                    <span className="vision-badge">Vision</span>
                  </Tooltip>
                }
              </Option>
            ))}
          </Select>
        </div>
        
        <div className="custom-model-container">
          <Checkbox 
            checked={useCustomModel} 
            onChange={handleCustomModelToggle}
          >
            Use custom model
          </Checkbox>
          
          {useCustomModel && (
            <div className="custom-url-input">
              <Input
                placeholder="Enter model URL or path (e.g., https://huggingface.co/model or local path)"
                value={customModelUrl}
                onChange={handleCustomUrlChange}
                suffix={
                  <Tooltip title="Enter a Hugging Face model URL, GitHub repository, or local path">
                    <InfoCircleOutlined style={{ color: 'rgba(0,0,0,.45)' }} />
                  </Tooltip>
                }
              />
            </div>
          )}
        </div>
        
        <div className="advanced-settings">
          <Button 
            type="link" 
            icon={<SettingOutlined />} 
            onClick={handleAdvancedSettingsToggle}
          >
            {advancedSettings ? 'Hide advanced settings' : 'Show advanced settings'}
          </Button>
          
          {advancedSettings && (
            <div className="advanced-settings-content">
              <div className="setting-item">
                <label>
                  Temperature: {temperature}
                  <Tooltip title="Controls randomness: lower values are more deterministic, higher values more creative">
                    <InfoCircleOutlined style={{ marginLeft: 8 }} />
                  </Tooltip>
                </label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => handleTemperatureChange(parseFloat(e.target.value))}
                  />
                </div>
              </div>
              
              <div className="setting-item">
                <label>
                  Max Tokens: {maxTokens}
                  <Tooltip title="Maximum length of the generated response">
                    <InfoCircleOutlined style={{ marginLeft: 8 }} />
                  </Tooltip>
                </label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="100"
                    max="4000"
                    step="100"
                    value={maxTokens}
                    onChange={(e) => handleMaxTokensChange(parseInt(e.target.value))}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className="model-selector-actions">
        <Button 
          type="primary" 
          onClick={saveSettings} 
          loading={loading}
          icon={<CloudDownloadOutlined />}
        >
          Save as Default
        </Button>
      </div>
    </div>
  );
};

export default ModelSelector; 