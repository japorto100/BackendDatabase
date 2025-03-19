import { html, css, LitElement } from 'lit';
import { repeat } from 'lit/directives/repeat.js';
import SearchAPI from '../api/search-api.js';
import { getProviderTemplatesList, getProviderTemplate } from './provider-templates.js';
import '../../css/provider-dialog.css';
import { getKGConnector } from '../api/kg-connector.js';

class SearchInterface extends LitElement {
    static properties = {
        mode: { type: String },
        query: { type: String },
        results: { type: Array },
        loading: { type: Boolean },
        error: { type: String },
        viewMode: { type: String }, // 'list', 'grid', 'split'
        providerFilters: { type: Object },
        timeRange: { type: String },
        sortBy: { type: String },
        language: { type: String },
        availableProviders: { type: Array },
        customProviders: { type: Array },
        selectedProviders: { type: Array },
        enhancedQuery: { type: String },
        isDeepResearch: { type: Boolean },
        deepResearchProgress: { type: Number },
        deepResearchStatus: { type: String },
        deepResearchFindings: { type: Array },
        searchImage: { type: Object }
    };

    constructor() {
        super();
        this.mode = 'all';
        this.query = '';
        this.results = [];
        this.loading = false;
        this.error = null;
        this.viewMode = 'list';
        this.timeRange = 'anytime';
        this.sortBy = 'relevance';
        this.language = 'en';
        this.isDeepResearch = false;
        this.deepResearchProgress = 0;
        this.deepResearchStatus = '';
        this.deepResearchFindings = [];
        this.searchImage = null;

        // Built-in providers
        this.availableProviders = [
            { 
                id: 'universal', 
                name: 'Universal Search', 
                icon: '🔍',
                description: 'AI-powered universal search across all sources',
                isDefault: true,
                filters: ['mode', 'depth', 'focus']
            },
            { 
                id: 'deep_research', 
                name: 'Deep Research', 
                icon: '🔬',
                description: 'In-depth exploration of topics with iterative research',
                filters: ['iterations', 'depth', 'breadth']
            },
            { 
                id: 'web', 
                name: 'Web', 
                icon: '🌐',
                filters: ['region', 'site', 'type']
            },
            { 
                id: 'academic', 
                name: 'Academic', 
                icon: '📚',
                filters: ['journal', 'year', 'citation_count', 'field']
            },
            { 
                id: 'youtube', 
                name: 'YouTube', 
                icon: '▶️',
                filters: ['duration', 'channel', 'quality', 'caption']
            },
            { 
                id: 'wolfram', 
                name: 'Wolfram', 
                icon: '🧮',
                filters: ['category', 'complexity', 'format']
            },
            { 
                id: 'reddit', 
                name: 'Reddit', 
                icon: '📱',
                filters: ['subreddit', 'sort', 'time', 'flair']
            },
            { 
                id: 'github', 
                name: 'GitHub', 
                icon: '💻',
                filters: ['language', 'stars', 'forks', 'updated']
            },
            { 
                id: 'docs', 
                name: 'Documentation', 
                icon: '📖',
                filters: ['source', 'type', 'framework', 'version', 'local']
            },
            {
                id: 'local_docs',
                name: 'Local Documents',
                icon: '📂',
                filters: ['file_type', 'folder', 'date_modified', 'content_type']
            },
            { 
                id: 'metabase', 
                name: 'Analytics', 
                icon: '📊',
                filters: [
                    'dashboard',
                    'chart_type',
                    'time_range',
                    'data_source',
                    'refresh_rate'
                ]
            },
            {
                id: 'eu_opendata',
                name: 'EU Data Portal',
                icon: '🇪🇺',
                filters: {
                    country: ['CH', 'DE', 'FR', 'IT', 'AT'],
                    dataType: ['company', 'economic', 'research', 'public'],
                    language: ['de', 'fr', 'it', 'en'],
                    year: 'all',
                    format: 'all'
                },
                baseUrl: 'https://data.europa.eu/api/hub/search/datasets'
            },
            {
                id: 'apollo',
                name: 'Apollo.io',
                icon: '🎯',
                filters: {
                    region: ['CH', 'DACH', 'EU'],
                    companySize: 'all',
                    industry: 'all',
                    technology: 'all',
                    jobTitle: 'all'
                },
                apiKey: process.env.APOLLO_API_KEY,
                baseUrl: 'https://api.apollo.io/v1'
            },
            {
                id: 'zefix',
                name: 'Zefix',  // Schweizer Handelsregister
                icon: '🇨🇭',
                filters: {
                    canton: 'all',
                    legalForm: 'all',
                    status: 'active',
                    year: 'all'
                }
            },
            {
                id: 'swissfirms',
                name: 'Swissfirms',
                icon: '🏢',
                filters: {
                    region: 'all',
                    industry: 'all',
                    size: 'all'
                }
            }
        ];

        // Custom providers added by user
        this.customProviders = [];
        
        // Default to universal search
        this.selectedProviders = ['universal'];

        this.providerFilters = {
            universal: {
                mode: 'smart',  // 'smart', 'focused', 'comprehensive'
                depth: 'auto',  // 'auto', 'shallow', 'deep'
                focus: 'all'    // 'all', 'academic', 'news', 'technical'
            },
            web: {
                region: 'global',
                site: '',
                type: 'all'
            },
            academic: {
                journal: '',
                year: 'all',
                citation_count: 0,
                field: 'all'
            },
            youtube: {
                duration: 'any',
                channel: '',
                quality: 'any',
                caption: false
            },
            wolfram: {
                category: 'all',
                complexity: 'medium',
                format: 'simple'
            },
            reddit: {
                subreddit: '',
                sort: 'relevance',
                time: 'all',
                flair: ''
            },
            github: {
                language: 'any',
                stars: 0,
                forks: 0,
                updated: 'anytime'
            },
            docs: {
                source: 'all',
                type: 'all',
                framework: 'all',
                version: 'latest',
                local: false
            },
            local_docs: {
                file_type: [
                    'pdf',
                    'doc',
                    'docx',
                    'txt',
                    'md',
                    'json',
                    'csv',
                    'xls',
                    'xlsx'
                ],
                folder: '/',
                date_modified: 'any',
                content_type: 'all'
            },
            metabase: {
                dashboard: 'all',        // specific dashboard or 'all'
                chart_type: 'any',      // table, line, bar, etc.
                time_range: 'auto',     // last_day, last_week, last_month, custom
                data_source: 'all',     // specific database or 'all'
                refresh_rate: 'auto'    // real-time, hourly, daily, etc.
            }
        };
    }

    renderProviderFilters(provider) {
        const filters = this.providerFilters[provider.id];
        if (!filters) return '';

        return html`
            <div class="provider-filters">
                <h4>${provider.name} Filters</h4>
                ${provider.filters.map(filterKey => {
                    switch(filterKey) {
                        case 'duration':
                            return html`
                                <select @change="${e => this.updateFilter(provider.id, 'duration', e.target.value)}">
                                    <option value="any">Any Length</option>
                                    <option value="short">< 4 minutes</option>
                                    <option value="medium">4-20 minutes</option>
                                    <option value="long">> 20 minutes</option>
                                </select>
                            `;
                        case 'year':
                            return html`
                                <select @change="${e => this.updateFilter(provider.id, 'year', e.target.value)}">
                                    <option value="all">All Years</option>
                                    <option value="2024">2024</option>
                                    <option value="2023">2023</option>
                                    <option value="last5">Last 5 Years</option>
                                    <option value="last10">Last 10 Years</option>
                                </select>
                            `;
                        case 'citation_count':
                            return html`
                                <input 
                                    type="number" 
                                    placeholder="Min Citations"
                                    @change="${e => this.updateFilter(provider.id, 'citation_count', e.target.value)}"
                                >
                            `;
                        // ... weitere Filter-Typen
                    }
                })}
            </div>
        `;
    }

    async search() {
        if (!this.query.trim() && !this.searchImage) return;
        
        this.loading = true;
        this.error = null;
        this.deepResearchFindings = [];
        this.deepResearchProgress = 0;

        try {
            // Check if Deep Research is selected
            if (this.selectedProviders.includes('deep_research')) {
                await this.performDeepResearch();
                return;
            }
            
            // Process searchImage if present
            let uploadedImagePath = null;
            if (this.searchImage) {
                try {
                    const formData = new FormData();
                    formData.append('image', this.searchImage);
                    
                    const uploadResponse = await fetch('/api/upload/', {
                        method: 'POST',
                        body: formData,
                        headers: {
                            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                        }
                    });
                    
                    if (!uploadResponse.ok) {
                        throw new Error('Failed to upload image');
                    }
                    
                    const uploadData = await uploadResponse.json();
                    uploadedImagePath = uploadData.file_paths?.[0] || null;
                } catch (error) {
                    console.error('Error uploading image:', error);
                    this.error = `Error uploading image: ${error.message}`;
                    this.loading = false;
                    return;
                }
            }
            
            // Determine if we should use multimodal search
            const isMultimodal = uploadedImagePath !== null;
            let searchEndpoint = '/api/search/';
            let requestBody = {
                query: this.query,
                providers: this.selectedProviders,
                filters: this.providerFilters
            };
            
            if (isMultimodal) {
                searchEndpoint = '/api/search/multimodal/';
                requestBody = {
                    query: this.query,
                    images: [uploadedImagePath],
                    providers: this.selectedProviders,
                    filters: this.providerFilters
                };
            } else {
                // Add KG enhancement if using the universal provider
                if (this.selectedProviders.includes('universal')) {
                    // Get the KG connector
                    const kgConnector = await this._getKGConnector();
                    
                    // Enhance the query with KG
                    if (kgConnector) {
                        this.enhancedQuery = await kgConnector.enhance_search_query(this.query);
                    } else {
                        this.enhancedQuery = this.query;
                    }
                } else {
                    this.enhancedQuery = this.query;
                }
                
                // Use enhanced query for search
                requestBody.query = this.enhancedQuery;
            }
            
            // Perform the search
            const response = await fetch(searchEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.status}`);
            }
            
            const data = await response.json();
            
            // For multimodal search, the enhanced query comes from the image understanding
            if (isMultimodal && data.enhanced_query) {
                this.enhancedQuery = data.enhanced_query;
            }
            
            this.results = data.results || [];
            
            // After search is complete, store valuable results in KG
            const kgConnector = await this._getKGConnector();
            if (kgConnector && this.results.length > 0) {
                kgConnector.store_search_results(this.query, this.results);
            }
        } catch (error) {
            console.error('Search error:', error);
            this.error = error.message;
        } finally {
            this.loading = false;
        }
    }

    sortResults(a, b) {
        switch(this.sortBy) {
            case 'relevance':
                return b.relevance - a.relevance;
            case 'date':
                return new Date(b.date) - new Date(a.date);
            case 'popularity':
                return b.popularity - a.popularity;
            default:
                return 0;
        }
    }

    toggleProvider(providerId) {
        if (this.selectedProviders.includes(providerId)) {
            this.selectedProviders = this.selectedProviders.filter(id => id !== providerId);
        } else {
            this.selectedProviders = [...this.selectedProviders, providerId];
        }
        this.requestUpdate();
    }

    setViewMode(mode) {
        this.viewMode = mode;
    }

    updateFilter(providerId, filterKey, value) {
        this.providerFilters[providerId][filterKey] = value;
    }

    renderMetabaseResult(result) {
        return html`
            <div class="metabase-result">
                <iframe
                    src="${result.embedUrl}"
                    frameborder="0"
                    width="100%"
                    height="${result.height || '400px'}"
                    allowtransparency
                ></iframe>
                <div class="result-controls">
                    <button @click="${() => this.refreshChart(result.id)}">
                        Refresh
                    </button>
                    <button @click="${() => this.exportChart(result.id)}">
                        Export
                    </button>
                </div>
            </div>
        `;
    }

    render() {
        return html`
            <div class="search-container">
                <div class="search-header">
                    <div class="search-input-container">
                        <input 
                            type="text" 
                            .value="${this.query}" 
                            @input="${e => this.query = e.target.value}"
                            @keydown="${e => e.key === 'Enter' && this.search()}"
                            placeholder="Search with Perplexica..."
                            ?disabled="${this.loading}"
                        />
                        <button @click="${this.search}" ?disabled="${this.loading}">
                            ${this.loading ? 'Searching...' : 'Search'}
                        </button>
                    </div>
                    
                    <div class="provider-selector">
                        ${this.renderProviderSelection()}
                    </div>
                </div>
                
                ${this.error ? html`<div class="error-message">${this.error}</div>` : ''}
                
                ${this.isDeepResearch && this.loading ? html`
                    <div class="deep-research-progress">
                        <h3>Deep Research in Progress: ${this.deepResearchStatus}</h3>
                        <progress value="${this.deepResearchProgress}" max="100"></progress>
                        <p>${this.deepResearchProgress}% complete</p>
                    </div>
                ` : ''}
                
                ${this.deepResearchFindings.length > 0 ? html`
                    <div class="deep-research-findings">
                        <h3>Research Findings</h3>
                        <div class="findings-container">
                            ${repeat(this.deepResearchFindings, (finding, i) => i, finding => html`
                                <div class="finding-card">
                                    <h4>${finding.concept}</h4>
                                    <p>${finding.summary}</p>
                                    ${finding.sources && finding.sources.length > 0 ? html`
                                        <div class="sources">
                                            <h5>Sources:</h5>
                                            <ul>
                                                ${repeat(finding.sources, (source, i) => i, source => html`
                                                    <li><a href="${source.url}" target="_blank">${source.title}</a></li>
                                                `)}
                                            </ul>
                                        </div>
                                    ` : ''}
                                </div>
                            `)}
                        </div>
                    </div>
                ` : ''}
                
                <div class="results-container ${this.viewMode}">
                    ${this.results.length > 0 ? html`
                        <div class="results-header">
                            <div class="results-count">
                                ${this.results.length} results ${this.enhancedQuery && this.enhancedQuery !== this.query ? 
                                    html`<span class="enhanced-query">(enhanced query: "${this.enhancedQuery}")</span>` : ''}
                            </div>
                            <div class="view-controls">
                                <button @click="${() => this.setViewMode('list')}" class="${this.viewMode === 'list' ? 'active' : ''}">List</button>
                                <button @click="${() => this.setViewMode('grid')}" class="${this.viewMode === 'grid' ? 'active' : ''}">Grid</button>
                                <button @click="${() => this.setViewMode('split')}" class="${this.viewMode === 'split' ? 'active' : ''}">Split</button>
                            </div>
                        </div>
                        
                        <div class="results-list">
                            ${repeat(this.results, result => result.id, result => this.renderResult(result))}
                        </div>
                    ` : this.loading ? html`
                        <div class="loading-indicator">
                            <div class="spinner"></div>
                            <p>Searching...</p>
                        </div>
                    ` : html`
                        <div class="no-results">
                            <p>No results found. Try a different search query.</p>
                        </div>
                    `}
                </div>
            </div>
        `;
    }

    renderProviderSelection() {
        return html`
            <div class="provider-selection">
                ${this.getAllProviders().map(provider => html`
                    <button 
                        class="provider-button ${this.selectedProviders.includes(provider.id) ? 'active' : ''}"
                        @click="${() => this.toggleProvider(provider.id)}">
                        <span class="icon">${provider.icon}</span>
                        ${provider.name}
                    </button>
                `)}
                <button class="add-provider" @click="${this._showAddProviderDialog}">
                    + Add Source
                </button>
            </div>
        `;
    }

    // Add new custom provider
    addCustomProvider(provider) {
        // Validiere Provider-Daten
        if (!provider.id || !provider.name) {
            console.error('Provider ID and name are required');
            return;
        }
        
        // Standardwerte setzen
        const newProvider = {
            id: provider.id,
            name: provider.name,
            icon: provider.icon || '🔍',
            filters: provider.filters || [],
            baseUrl: provider.baseUrl || '',
            apiKey: provider.apiKey || '',
            customHeaders: provider.customHeaders || {},
            providerType: provider.providerType || 'custom',
            config: provider.config || {}
        };
        
        // Provider-spezifische Konfiguration
        if (provider.providerType === 'database') {
            newProvider.config.database_url = provider.databaseUrl;
        } else if (provider.providerType === 'graphql') {
            newProvider.config.graphql_endpoint = provider.graphqlEndpoint;
        } else if (provider.providerType === 'filesystem') {
            newProvider.config.filesystem_path = provider.filesystemPath;
        } else if (provider.providerType === 'streaming') {
            newProvider.config.streaming_endpoint = provider.streamingEndpoint;
        } else if (provider.providerType === 'enterprise') {
            // LDAP, FTP, etc.
            Object.assign(newProvider.config, provider.enterpriseConfig || {});
        }
        
        this.customProviders = [...this.customProviders, newProvider];
        
        // Füge Provider zur Datenbank hinzu
        this._saveProviderToDatabase(newProvider);
    }

    // Speichere Provider in der Datenbank
    async _saveProviderToDatabase(provider) {
        try {
            const response = await fetch('/api/providers/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this._getCsrfToken()
                },
                body: JSON.stringify({
                    name: provider.name,
                    provider_type: provider.providerType,
                    api_key: provider.apiKey,
                    base_url: provider.baseUrl,
                    custom_headers: provider.customHeaders,
                    config: provider.config,
                    is_active: true
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to save provider');
            }
            
            const data = await response.json();
            console.log('Provider saved:', data);
        } catch (error) {
            console.error('Error saving provider:', error);
        }
    }

    // CSRF Token aus Cookie holen
    _getCsrfToken() {
        const name = 'csrftoken=';
        const decodedCookie = decodeURIComponent(document.cookie);
        const cookieArray = decodedCookie.split(';');
        
        for (let i = 0; i < cookieArray.length; i++) {
            let cookie = cookieArray[i].trim();
            if (cookie.indexOf(name) === 0) {
                return cookie.substring(name.length, cookie.length);
            }
        }
        
        return '';
    }

    // Dialog zum Hinzufügen eines neuen Providers
    _showAddProviderDialog() {
        const dialog = document.createElement('dialog');
        dialog.className = 'provider-dialog';
        
        // Hole Provider-Vorlagen
        const templates = getProviderTemplatesList();
        
        dialog.innerHTML = `
            <div class="provider-dialog-content">
                <h2>Provider hinzufügen</h2>
                
                <div class="provider-tabs">
                    <button class="tab-btn active" data-tab="template">Vorlage verwenden</button>
                    <button class="tab-btn" data-tab="custom">Benutzerdefiniert</button>
                    <button class="tab-btn" data-tab="url">Von URL erkennen</button>
                </div>
                
                <div class="tab-content" id="template-tab">
                    <div class="template-list">
                        ${templates.map(template => `
                            <div class="template-item" data-id="${template.id}">
                                <div class="template-icon">${template.icon}</div>
                                <div class="template-info">
                                    <h3>${template.name}</h3>
                                    <p>${template.description}</p>
                                    ${template.requiresApiKey ? 
                                        '<span class="api-key-required">API-Key erforderlich</span>' : 
                                        '<span class="no-api-key">Kein API-Key erforderlich</span>'
                                    }
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="tab-content hidden" id="custom-tab">
                    <form id="custom-provider-form">
                        <div class="form-group">
                            <label for="provider-id">ID</label>
                            <input type="text" id="provider-id" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="provider-name">Name</label>
                            <input type="text" id="provider-name" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="provider-icon">Icon</label>
                            <input type="text" id="provider-icon" value="🔍">
                        </div>
                        
                        <div class="form-group">
                            <label for="provider-type">Provider Type</label>
                            <select id="provider-type">
                                <option value="web">Web Scraping</option>
                                <option value="api">REST API</option>
                                <option value="graphql">GraphQL API</option>
                                <option value="database">Database</option>
                                <option value="filesystem">Filesystem</option>
                                <option value="streaming">Streaming</option>
                                <option value="enterprise">Enterprise (LDAP/FTP)</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="provider-url">Base URL</label>
                            <input type="url" id="provider-url">
                        </div>
                        
                        <div class="form-group">
                            <label for="provider-api-key">API Key</label>
                            <input type="password" id="provider-api-key">
                        </div>
                        
                        <div class="form-group">
                            <label for="provider-headers">Custom Headers (JSON)</label>
                            <textarea id="provider-headers" rows="3"></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label for="provider-config">Additional Config (JSON)</label>
                            <textarea id="provider-config" rows="5"></textarea>
                        </div>
                    </form>
                </div>
                
                <div class="tab-content hidden" id="url-tab">
                    <div class="url-detection-form">
                        <div class="form-group">
                            <label for="detect-url">URL zur Analyse</label>
                            <input type="url" id="detect-url" placeholder="https://example.com">
                        </div>
                        
                        <button id="detect-btn" type="button">Provider-Typ erkennen</button>
                        
                        <div id="detection-result" class="detection-result hidden">
                            <h3>Erkannter Provider-Typ: <span id="detected-type">-</span></h3>
                            <div id="detection-details"></div>
                        </div>
                    </div>
                </div>
                
                <div class="dialog-actions">
                    <button id="cancel-btn" type="button">Abbrechen</button>
                    <button id="save-btn" type="button">Provider hinzufügen</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(dialog);
        dialog.showModal();
        
        // Tab-Wechsel
        const tabBtns = dialog.querySelectorAll('.tab-btn');
        const tabContents = dialog.querySelectorAll('.tab-content');
        
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;
                
                // Tabs umschalten
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Inhalte umschalten
                tabContents.forEach(content => {
                    content.classList.add('hidden');
                });
                dialog.querySelector(`#${tabId}-tab`).classList.remove('hidden');
            });
        });
        
        // Vorlage auswählen
        const templateItems = dialog.querySelectorAll('.template-item');
        templateItems.forEach(item => {
            item.addEventListener('click', () => {
                templateItems.forEach(i => i.classList.remove('selected'));
                item.classList.add('selected');
            });
        });
        
        // Provider-Typ erkennen
        const detectBtn = dialog.querySelector('#detect-btn');
        detectBtn.addEventListener('click', async () => {
            const url = dialog.querySelector('#detect-url').value;
            if (!url) return;
            
            detectBtn.disabled = true;
            detectBtn.textContent = 'Erkenne...';
            
            try {
                const result = await this._detectProviderType(url);
                
                dialog.querySelector('#detected-type').textContent = result.providerType;
                
                const detailsDiv = dialog.querySelector('#detection-details');
                detailsDiv.innerHTML = `
                    <div class="detection-config">
                        <h4>Erkannte Konfiguration:</h4>
                        <pre>${JSON.stringify(result.config || {}, null, 2)}</pre>
                    </div>
                    
                    ${result.apiKeyRequired ? 
                        '<div class="api-key-notice">Dieser Provider benötigt einen API-Key</div>' : 
                        ''
                    }
                    
                    <button id="use-detection-btn" type="button">Diese Konfiguration verwenden</button>
                `;
                
                dialog.querySelector('#detection-result').classList.remove('hidden');
                
                // Erkannte Konfiguration verwenden
                dialog.querySelector('#use-detection-btn').addEventListener('click', () => {
                    // Wechsle zum benutzerdefinierten Tab
                    tabBtns.forEach(b => b.classList.remove('active'));
                    dialog.querySelector('[data-tab="custom"]').classList.add('active');
                    
                    tabContents.forEach(content => {
                        content.classList.add('hidden');
                    });
                    dialog.querySelector('#custom-tab').classList.remove('hidden');
                    
                    // Fülle das Formular mit erkannten Werten
                    const urlParts = new URL(url);
                    const domain = urlParts.hostname.replace('www.', '');
                    
                    dialog.querySelector('#provider-id').value = domain.split('.')[0];
                    dialog.querySelector('#provider-name').value = 
                        domain.charAt(0).toUpperCase() + domain.slice(1).split('.')[0];
                    dialog.querySelector('#provider-type').value = result.providerType;
                    dialog.querySelector('#provider-url').value = url;
                    
                    if (result.config) {
                        dialog.querySelector('#provider-config').value = 
                            JSON.stringify(result.config, null, 2);
                    }
                });
            } catch (error) {
                console.error('Error detecting provider type:', error);
                dialog.querySelector('#detection-result').innerHTML = `
                    <div class="error-message">
                        Fehler bei der Erkennung: ${error.message}
                    </div>
                `;
                dialog.querySelector('#detection-result').classList.remove('hidden');
            } finally {
                detectBtn.disabled = false;
                detectBtn.textContent = 'Provider-Typ erkennen';
            }
        });
        
        // Abbrechen
        dialog.querySelector('#cancel-btn').addEventListener('click', () => {
            dialog.close();
            dialog.remove();
        });
        
        // Provider hinzufügen
        dialog.querySelector('#save-btn').addEventListener('click', () => {
            const activeTab = dialog.querySelector('.tab-btn.active').dataset.tab;
            
            if (activeTab === 'template') {
                // Provider aus Vorlage erstellen
                const selectedTemplate = dialog.querySelector('.template-item.selected');
                if (!selectedTemplate) {
                    alert('Bitte wählen Sie eine Vorlage aus');
                    return;
                }
                
                const templateId = selectedTemplate.dataset.id;
                const template = getProviderTemplate(templateId);
                
                // Wenn API-Key erforderlich ist, nach diesem fragen
                if (template.requiresApiKey) {
                    const apiKey = prompt(`Bitte geben Sie den API-Key für ${template.name} ein:`);
                    if (!apiKey) {
                        alert('API-Key ist erforderlich');
                        return;
                    }
                    
                    // Provider mit API-Key erstellen
                    this.addCustomProvider({
                        id: templateId,
                        name: template.name,
                        icon: template.icon,
                        providerType: template.providerType,
                        baseUrl: template.baseUrl,
                        apiKey: apiKey,
                        config: template.defaultConfig
                    });
                } else {
                    // Provider ohne API-Key erstellen
                    this.addCustomProvider({
                        id: templateId,
                        name: template.name,
                        icon: template.icon,
                        providerType: template.providerType,
                        baseUrl: template.baseUrl,
                        config: template.defaultConfig
                    });
                }
            } else if (activeTab === 'custom') {
                // Benutzerdefinierten Provider erstellen
                const form = dialog.querySelector('#custom-provider-form');
                
                const provider = {
                    id: form.querySelector('#provider-id').value,
                    name: form.querySelector('#provider-name').value,
                    icon: form.querySelector('#provider-icon').value,
                    providerType: form.querySelector('#provider-type').value,
                    baseUrl: form.querySelector('#provider-url').value,
                    apiKey: form.querySelector('#provider-api-key').value
                };
                
                // Custom Headers parsen
                try {
                    const headersText = form.querySelector('#provider-headers').value;
                    if (headersText) {
                        provider.customHeaders = JSON.parse(headersText);
                    }
                } catch (error) {
                    alert('Ungültiges JSON-Format in Custom Headers');
                    return;
                }
                
                // Config parsen
                try {
                    const configText = form.querySelector('#provider-config').value;
                    if (configText) {
                        provider.config = JSON.parse(configText);
                    }
                } catch (error) {
                    alert('Ungültiges JSON-Format in Config');
                    return;
                }
                
                this.addCustomProvider(provider);
            } else if (activeTab === 'url') {
                // Provider aus erkannter URL erstellen
                const detectionResult = dialog.querySelector('#detection-result');
                if (detectionResult.classList.contains('hidden')) {
                    alert('Bitte führen Sie zuerst eine Erkennung durch');
                    return;
                }
                
                const url = dialog.querySelector('#detect-url').value;
                const urlParts = new URL(url);
                const domain = urlParts.hostname.replace('www.', '');
                const providerType = dialog.querySelector('#detected-type').textContent;
                
                // Config aus dem Pre-Element extrahieren
                let config = {};
                try {
                    const configText = dialog.querySelector('.detection-config pre').textContent;
                    if (configText) {
                        config = JSON.parse(configText);
                    }
                } catch (error) {
                    console.error('Error parsing config:', error);
                }
                
                // API-Key abfragen, wenn erforderlich
                let apiKey = '';
                if (dialog.querySelector('.api-key-notice')) {
                    apiKey = prompt(`Bitte geben Sie den API-Key für ${domain} ein:`);
                    if (!apiKey) {
                        alert('API-Key ist erforderlich');
                        return;
                    }
                }
                
                this.addCustomProvider({
                    id: domain.split('.')[0],
                    name: domain.charAt(0).toUpperCase() + domain.slice(1).split('.')[0],
                    icon: '🔍',
                    providerType: providerType,
                    baseUrl: url,
                    apiKey: apiKey,
                    config: config
                });
            }
            
            dialog.close();
            dialog.remove();
        });
    }

    // Get all available providers (built-in + custom)
    getAllProviders() {
        return [...this.availableProviders, ...this.customProviders];
    }

    // Füge diese neue Methode hinzu
    async _detectProviderType(url) {
        this.loading = true;
        try {
            const response = await fetch('/api/detect-provider-type/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this._getCsrfToken()
                },
                body: JSON.stringify({ url })
            });
            
            if (!response.ok) {
                throw new Error('Failed to detect provider type');
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error detecting provider type:', error);
            return { providerType: 'web' }; // Fallback auf Web Scraping
        } finally {
            this.loading = false;
        }
    }

    // Füge diese Methode hinzu
    async _validateApiKey(providerType, baseUrl, apiKey) {
        try {
            const response = await fetch('/api/validate-api-key/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this._getCsrfToken()
                },
                body: JSON.stringify({
                    provider_type: providerType,
                    base_url: baseUrl,
                    api_key: apiKey
                })
            });
            
            const data = await response.json();
            return {
                valid: response.ok,
                message: data.message || 'API-Key validiert'
            };
        } catch (error) {
            return {
                valid: false,
                message: 'Fehler bei der Validierung: ' + error.message
            };
        }
    }

    // Add method to get KG connector
    async _getKGConnector() {
        try {
            const response = await fetch('/api/kg-connector/');
            if (!response.ok) return null;
            
            return await response.json();
        } catch (error) {
            console.error('Error getting KG connector:', error);
            return null;
        }
    }

    async performDeepResearch() {
        this.isDeepResearch = true;
        this.deepResearchStatus = 'Starting deep research...';
        
        try {
            // Get filters for deep research
            const filters = this.providerFilters['deep_research'] || {
                iterations: 3,
                depth: 'medium',
                breadth: 'balanced'
            };
            
            // Initialize deep research
            const response = await fetch('/api/search/deep-research/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this._getCsrfToken()
                },
                body: JSON.stringify({
                    query: this.query,
                    max_iterations: filters.iterations,
                    depth: filters.depth,
                    breadth: filters.breadth
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to start deep research');
            }
            
            const data = await response.json();
            const researchId = data.research_id;
            
            // Poll for progress
            await this.pollDeepResearchProgress(researchId);
            
            // Get final results
            const resultsResponse = await fetch(`/api/search/deep-research/${researchId}/results`);
            if (!resultsResponse.ok) {
                throw new Error('Failed to retrieve deep research results');
            }
            
            const resultsData = await resultsResponse.json();
            
            // Process and display results
            this.deepResearchFindings = resultsData.findings || [];
            this.results = resultsData.results || [];
            
            // Update status
            this.deepResearchStatus = 'Research complete';
            this.deepResearchProgress = 100;
        } catch (error) {
            this.error = `Deep Research error: ${error.message}`;
            this.deepResearchStatus = 'Research failed';
        } finally {
            this.loading = false;
        }
    }
    
    async pollDeepResearchProgress(researchId) {
        return new Promise((resolve, reject) => {
            const checkProgress = async () => {
                try {
                    const response = await fetch(`/api/search/deep-research/${researchId}/status`);
                    if (!response.ok) {
                        reject(new Error('Failed to check research progress'));
                        return;
                    }
                    
                    const data = await response.json();
                    this.deepResearchProgress = data.progress;
                    this.deepResearchStatus = data.status;
                    
                    if (data.status === 'completed' || data.status === 'failed') {
                        resolve();
                    } else {
                        setTimeout(checkProgress, 2000); // Check every 2 seconds
                    }
                } catch (error) {
                    reject(error);
                }
            };
            
            checkProgress();
        });
    }

    setupImageSearchCapability() {
        // Add image search icon next to search input
        const searchContainer = this.shadowRoot.querySelector('.search-input-container');
        
        // Create image upload button
        const imageUploadBtn = document.createElement('button');
        imageUploadBtn.className = 'image-search-btn';
        imageUploadBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z"></path><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>';
        imageUploadBtn.title = 'Search with Image';
        
        // Create hidden file input
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        fileInput.id = 'image-search-input';
        
        // Add to DOM
        searchContainer.insertBefore(imageUploadBtn, searchContainer.firstChild);
        searchContainer.appendChild(fileInput);
        
        // Setup event listeners
        imageUploadBtn.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', this.handleImageSearch.bind(this));
        
        // Container for image preview
        this.imagePreviewContainer = document.createElement('div');
        this.imagePreviewContainer.className = 'search-image-preview';
        this.imagePreviewContainer.style.display = 'none';
        searchContainer.parentNode.insertBefore(this.imagePreviewContainer, searchContainer.nextSibling);
        
        // Initialize searchImage property
        this.searchImage = null;
    }
    
    handleImageSearch(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Validate file type
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }
        
        // Store the file
        this.searchImage = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.imagePreviewContainer.innerHTML = '';
            
            const previewWrapper = document.createElement('div');
            previewWrapper.className = 'preview-wrapper';
            
            // Image preview
            const image = document.createElement('img');
            image.src = e.target.result;
            image.className = 'search-image-preview-img';
            previewWrapper.appendChild(image);
            
            // Remove button
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-search-image';
            removeBtn.innerHTML = '×';
            removeBtn.addEventListener('click', () => {
                this.searchImage = null;
                this.imagePreviewContainer.style.display = 'none';
            });
            previewWrapper.appendChild(removeBtn);
            
            this.imagePreviewContainer.appendChild(previewWrapper);
            this.imagePreviewContainer.style.display = 'block';
        };
        
        reader.readAsDataURL(file);
        event.target.value = '';
    }

    static styles = css`
        .search-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            padding: 1rem;
        }

        .search-header {
            position: sticky;
            top: 0;
            background: var(--search-bg, white);
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .search-input-container {
            display: flex;
            gap: 0.5rem;
        }

        .search-input-container input {
            flex: 1;
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }

        .provider-selector {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .provider-btn {
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            cursor: pointer;
        }

        .provider-btn.active {
            background: var(--primary-color);
            color: white;
        }

        .results-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .results-container.grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        }

        .results-container.split {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        .result-item {
            padding: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }

        .result-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.8em;
            color: var(--text-secondary);
        }

        .global-filters {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
            padding: 1rem;
            background: var(--filter-bg, #f5f5f5);
            border-radius: 4px;
        }

        .provider-filters {
            margin: 0.5rem 0;
            padding: 1rem;
            background: var(--provider-filter-bg, #fff);
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }

        select, input {
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: white;
        }

        .deep-research-progress {
            margin: 20px 0;
            padding: 15px;
            background-color: #f5f9ff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .deep-research-progress progress {
            width: 100%;
            height: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .deep-research-findings {
            margin: 20px 0;
        }
        
        .findings-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .finding-card {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 15px;
            transition: transform 0.2s ease;
        }
        
        .finding-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .finding-card h4 {
            margin-top: 0;
            color: #2a5885;
        }
        
        .finding-card .sources {
            margin-top: 10px;
            font-size: 0.9em;
        }
        
        .finding-card .sources ul {
            padding-left: 20px;
        }
        
        .enhanced-query {
            font-style: italic;
            color: #666;
            font-size: 0.9em;
            margin-left: 10px;
        }

        .image-search-btn {
            background: transparent;
            border: none;
            cursor: pointer;
            padding: 8px;
            border-radius: 4px;
            margin-right: 8px;
            color: var(--search-icon-color, #666);
        }
        
        .image-search-btn:hover {
            background: var(--search-btn-hover-bg, rgba(0,0,0,0.05));
        }
        
        .search-image-preview {
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
            background: var(--search-preview-bg, #f5f5f5);
        }
        
        .preview-wrapper {
            position: relative;
            display: inline-block;
            max-width: 200px;
        }
        
        .search-image-preview-img {
            max-width: 100%;
            max-height: 150px;
            border-radius: 4px;
            border: 1px solid var(--search-border-color, #ddd);
        }
        
        .remove-search-image {
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(0,0,0,0.6);
            color: white;
            border: none;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    `;

    connectedCallback() {
        super.connectedCallback();
        
        // Setup after the component is connected
        setTimeout(() => {
            this.setupImageSearchCapability();
        }, 0);
    }
}

customElements.define('perplexica-search-interface', SearchInterface);

export default SearchInterface; 