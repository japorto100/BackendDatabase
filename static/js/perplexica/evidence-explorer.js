/**
 * Evidence Explorer Component
 * 
 * Zeigt Quelldokumente und Zitate für KI-generierte Antworten an.
 * Unterstützt jetzt auch @-Mentions als explizite Evidenzquellen.
 */

import { html, css, LitElement } from 'lit';
import './file-preview.js';

export class EvidenceExplorer extends LitElement {
    static properties = {
        responseId: { type: String },
        responseContent: { type: String },
        evidenceSources: { type: Array },
        activeSourceIndex: { type: Number },
        isExpanded: { type: Boolean },
        showConfidence: { type: Boolean },
        layout: { type: String },
        queryId: { type: String },
        isLoading: { type: Boolean },
        error: { type: String },
        sources: { type: Array },
        activeCitation: { type: Number },
        activeSource: { type: Object },
        isOpen: { type: Boolean },
        mentionSources: { type: Array },
        loading: { type: Boolean },
        kgEntities: { type: Array },
        showKgExplorer: { type: Boolean }
    };

    constructor() {
        super();
        this.responseId = '';
        this.responseContent = '';
        this.evidenceSources = [];
        this.activeSourceIndex = 0;
        this.isExpanded = true;
        this.showConfidence = false;
        this.layout = 'split'; // 'split', 'top-bottom', or 'single'
        
        this.queryId = '';
        this.isLoading = false;
        this.error = null;
        this.sources = [];
        this.mentionSources = [];
        this.activeCitation = null;
        this.activeSource = null;
        this.isOpen = false;
        this.loading = false;
        this.kgEntities = [];
        this.showKgExplorer = false;
    }

    connectedCallback() {
        super.connectedCallback();
        this.addEventListener('citation-click', this.handleCitationClick);
        
        if (this.queryId) {
            this.fetchEvidence();
        }
        
        // Event-Listener für Mention-Events hinzufügen
        window.addEventListener('mention-selected', this.handleMentionSelected.bind(this));
        document.addEventListener('citation-clicked', this.handleCitationClicked.bind(this));
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.removeEventListener('citation-click', this.handleCitationClick);
        
        // Event-Listener entfernen
        window.removeEventListener('mention-selected', this.handleMentionSelected.bind(this));
        document.removeEventListener('citation-clicked', this.handleCitationClicked.bind(this));
    }
    
    updated(changedProperties) {
        if (changedProperties.has('queryId') && this.queryId) {
            this.fetchEvidence();
        }
    }
    
    async fetchEvidence() {
        if (!this.queryId) return;
        
        this.isLoading = true;
        this.error = null;
        
        try {
            const response = await fetch(`/models/api/evidence/?query_id=${encodeURIComponent(this.queryId)}`);
            if (!response.ok) {
                throw new Error(`Failed to load evidence: ${response.statusText}`);
            }
            
            const data = await response.json();
            const results = data.results || data;
            
            this.evidenceSources = results.map((item, index) => ({
                id: item.id,
                name: `Source ${index + 1}`,
                type: item.source_type,
                content: item.content,
                path: item.id,
                highlights: this.convertHighlights(item.highlights)
            }));
            
            // Fetch related KG entities if available
            await this.fetchRelatedKgEntities();
            
        } catch (error) {
            console.error('Error loading evidence:', error);
            this.error = error.message;
        } finally {
            this.isLoading = false;
        }
    }
    
    async fetchRelatedKgEntities() {
        if (!this.queryId) return;
        
        try {
            const response = await fetch(`/api/kg/entities/?query=${encodeURIComponent(this.queryId)}`);
            if (!response.ok) {
                console.warn('Failed to load KG entities');
                return;
            }
            
            const data = await response.json();
            this.kgEntities = data.entities || [];
            
            // Add KG entities as evidence sources if they exist
            if (this.kgEntities.length > 0) {
                const kgSources = this.kgEntities.map((entity, index) => ({
                    id: `kg-${entity.id}`,
                    name: `KG: ${entity.label || `Entity ${index + 1}`}`,
                    type: 'knowledge_graph',
                    entity: entity,
                    relationships: entity.relationships || [],
                    content: JSON.stringify(entity, null, 2),
                    path: `kg-entity-${entity.id}`,
                    highlights: []
                }));
                
                this.evidenceSources = [...this.evidenceSources, ...kgSources];
            }
        } catch (error) {
            console.error('Error loading KG entities:', error);
        }
    }
    
    convertHighlights(apiHighlights) {
        if (!apiHighlights || !Array.isArray(apiHighlights)) return [];
        
        return apiHighlights.map((h, index) => ({
            id: `highlight-${index}`,
            citationId: `citation-${index}`,
            start: h.start,
            end: h.end,
            text: h.text,
            confidence: h.confidence || 0.5
        }));
    }

    handleCitationClick(e) {
        const citationId = e.detail.citationId;
        const source = this.evidenceSources.find(source => 
            source.highlights.some(h => h.citationId === citationId)
        );
        
        if (source) {
            const sourceIndex = this.evidenceSources.indexOf(source);
            this.activeSourceIndex = sourceIndex;
            
            const highlight = source.highlights.find(h => h.citationId === citationId);
            if (highlight) {
                this.requestUpdate().then(() => {
                    this.scrollToHighlight(highlight.id);
                });
            }
        }
    }

    scrollToHighlight(highlightId) {
        const highlightEl = this.shadowRoot.querySelector(`[data-id="${highlightId}"]`);
        if (highlightEl) {
            highlightEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            highlightEl.classList.add('flash-highlight');
            setTimeout(() => {
                highlightEl.classList.remove('flash-highlight');
            }, 1500);
        }
    }

    toggleExpand() {
        this.isExpanded = !this.isExpanded;
    }

    toggleConfidence() {
        this.showConfidence = !this.showConfidence;
    }

    changeLayout(layout) {
        this.layout = layout;
    }

    selectSource(index) {
        this.activeSourceIndex = index;
    }

    renderActiveSource() {
        if (this.isLoading) {
            return html`<div class="loading">Loading evidence sources...</div>`;
        }
        
        if (this.error) {
            return html`<div class="error">${this.error}</div>`;
        }
        
        if (this.evidenceSources.length === 0) {
            return html`<div class="no-evidence">No evidence sources available</div>`;
        }

        const source = this.evidenceSources[this.activeSourceIndex];
        
        // Check if this is a knowledge graph source
        if (source.type === 'knowledge_graph') {
            return html`
                <div class="kg-source">
                    <h4>Knowledge Graph Entity: ${source.entity?.label || 'Unknown'}</h4>
                    
                    <div class="kg-entity-details">
                        <div class="kg-entity-id">
                            <strong>ID:</strong> ${source.entity?.id || 'Unknown'}
                        </div>
                        
                        <div class="kg-entity-type">
                            <strong>Type:</strong> ${source.entity?.type || 'Unknown'}
                        </div>
                        
                        ${source.entity?.description ? html`
                            <div class="kg-entity-description">
                                <strong>Description:</strong> ${source.entity.description}
                            </div>
                        ` : ''}
                    </div>
                    
                    <div class="kg-properties">
                        <h5>Properties</h5>
                        ${source.entity?.properties && Object.keys(source.entity.properties).length > 0 ? 
                          html`<div class="properties-grid">
                            ${Object.entries(source.entity.properties).map(([key, value]) => html`
                              <div class="property-row">
                                  <div class="property-key">${key}:</div>
                                  <div class="property-value">${typeof value === 'object' ? JSON.stringify(value) : value}</div>
                              </div>
                            `)}
                          </div>` : 
                          html`<div class="no-data">No properties available</div>`
                        }
                    </div>
                    
                    <div class="kg-relationships">
                        <h5>Relationships</h5>
                        ${source.relationships && source.relationships.length > 0 ? 
                          html`<div class="relationships-list">
                            ${source.relationships.map(rel => html`
                              <div class="relationship-item" @click=${() => this.navigateToEntity(rel.target_id)}>
                                  <div class="relationship-source">${rel.source_label || rel.source}</div>
                                  <div class="relationship-type">${rel.type}</div>
                                  <div class="relationship-target">${rel.target_label || rel.target}</div>
                                  <div class="relationship-arrow">→</div>
                              </div>
                            `)}
                          </div>` : 
                          html`<div class="no-data">No relationships found</div>`
                        }
                    </div>
                    
                    <div class="kg-sources">
                        <h5>Sources</h5>
                        ${source.entity?.sources && source.entity.sources.length > 0 ? 
                          html`<ul class="sources-list">
                            ${source.entity.sources.map(src => html`
                              <li class="source-item">
                                  <a href="${src.url}" target="_blank">${src.title || src.url}</a>
                                  ${src.date ? html`<span class="source-date">${src.date}</span>` : ''}
                              </li>
                            `)}
                          </ul>` : 
                          html`<div class="no-data">No sources available</div>`
                        }
                    </div>
                    
                    <div class="kg-actions">
                        <button @click=${() => this.toggleKgExplorer()} class="kg-explore-btn">
                            ${this.showKgExplorer ? 'Hide Graph Explorer' : 'Explore in Graph View'}
                        </button>
                    </div>
                    
                    ${this.showKgExplorer ? html`
                        <div class="kg-explorer">
                            <h5>Knowledge Graph Explorer</h5>
                            <div class="kg-graph-container" id="kg-graph-${source.entity?.id}"></div>
                        </div>
                    ` : ''}
                </div>
            `;
        }
        
        // Default rendering for other source types
        return html`
            <file-preview
                filePath=${source.path}
                fileName=${source.name}
                fileContent=${source.content}
                fileType=${source.type}
                .highlights=${source.highlights}
                .showConfidence=${this.showConfidence}
                isExpanded=${true}
            ></file-preview>
        `;
    }

    renderSourceTabs() {
        return html`
            <div class="source-tabs">
                ${this.evidenceSources.map((source, index) => html`
                    <button 
                        class="source-tab ${index === this.activeSourceIndex ? 'active' : ''}"
                        @click=${() => this.selectSource(index)}
                    >
                        ${source.name}
                    </button>
                `)}
            </div>
        `;
    }

    renderOptionsBar() {
        return html`
            <div class="options-bar">
                <div class="layout-controls">
                    <button 
                        class="layout-btn ${this.layout === 'split' ? 'active' : ''}" 
                        @click=${() => this.changeLayout('split')}
                        title="Split View"
                    >◧</button>
                    <button 
                        class="layout-btn ${this.layout === 'top-bottom' ? 'active' : ''}" 
                        @click=${() => this.changeLayout('top-bottom')}
                        title="Stacked View"
                    >⬓</button>
                    <button 
                        class="layout-btn ${this.layout === 'single' ? 'active' : ''}" 
                        @click=${() => this.changeLayout('single')}
                        title="Response Only View"
                    >≡</button>
                </div>
                
                <button 
                    class="confidence-toggle ${this.showConfidence ? 'active' : ''}"
                    @click=${this.toggleConfidence}
                    title="Toggle Confidence Highlighting"
                >
                    ${this.showConfidence ? '✓ Show Confidence' : 'Show Confidence'}
                </button>
                
                <button 
                    class="expand-toggle"
                    @click=${this.toggleExpand}
                    title="Toggle Expand/Collapse"
                >
                    ${this.isExpanded ? 'Collapse' : 'Expand'}
                </button>
            </div>
        `;
    }

    formatResponseWithCitations() {
        return html`<div class="response-content">${this.responseContent}</div>`;
    }

    render() {
        if (!this.isExpanded) {
            return html`
                <div class="evidence-explorer collapsed">
                    <div class="header" @click=${this.toggleExpand}>
                        <h3>Evidence Explorer</h3>
                        <span class="toggle-icon">▶</span>
                    </div>
                </div>
            `;
        }

        if (this.layout === 'single') {
            return html`
                <div class="evidence-explorer expanded single-view">
                    ${this.renderOptionsBar()}
                    <div class="content-container">
                        <div class="response-panel full-width">
                            <h3>AI Response</h3>
                            ${this.formatResponseWithCitations()}
                        </div>
                    </div>
                </div>
            `;
        }

        if (this.layout === 'top-bottom') {
            return html`
                <div class="evidence-explorer expanded stacked-view">
                    ${this.renderOptionsBar()}
                    <div class="content-container">
                        <div class="response-panel">
                            <h3>AI Response</h3>
                            ${this.formatResponseWithCitations()}
                        </div>
                        <div class="evidence-panel">
                            <h3>Evidence Sources</h3>
                            ${this.renderSourceTabs()}
                            ${this.renderActiveSource()}
                        </div>
                    </div>
                </div>
            `;
        }

        return html`
            <div class="evidence-explorer expanded split-view">
                ${this.renderOptionsBar()}
                <div class="content-container">
                    <div class="response-panel">
                        <h3>AI Response</h3>
                        ${this.formatResponseWithCitations()}
                    </div>
                    <div class="evidence-panel">
                        <h3>Evidence Sources</h3>
                        ${this.renderSourceTabs()}
                        ${this.renderActiveSource()}
                    </div>
                </div>
            </div>
        `;
    }

    static styles = css`
        :host {
            display: block;
            font-family: var(--font-family, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif);
        }
        
        .evidence-explorer {
            border: 1px solid var(--border-color, #ddd);
            border-radius: 8px;
            overflow: hidden;
            background-color: var(--bg-color, #f8f8f8);
            margin-bottom: 16px;
        }
        
        .evidence-explorer.collapsed {
            cursor: pointer;
        }
        
        .evidence-explorer.collapsed:hover {
            background-color: var(--hover-bg, #f0f0f0);
        }
        
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background-color: var(--header-bg, #2d2d2d);
            color: var(--header-color, #e0e0e0);
        }
        
        .header h3 {
            margin: 0;
            font-size: 16px;
        }
        
        .options-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background-color: var(--options-bg, #2d2d2d);
            border-bottom: 1px solid var(--border-color, #333);
        }
        
        .layout-controls {
            display: flex;
            gap: 8px;
        }
        
        .layout-btn, .confidence-toggle, .expand-toggle {
            background-color: var(--btn-bg, #3d3d3d);
            border: none;
            color: var(--btn-color, #e0e0e0);
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .layout-btn:hover, .confidence-toggle:hover, .expand-toggle:hover {
            background-color: var(--btn-hover-bg, #4d4d4d);
        }
        
        .layout-btn.active, .confidence-toggle.active {
            background-color: var(--btn-active-bg, #5d5d5d);
        }
        
        .content-container {
            display: flex;
            height: 600px;
        }
        
        .split-view .content-container {
            flex-direction: row;
        }
        
        .stacked-view .content-container {
            flex-direction: column;
        }
        
        .response-panel, .evidence-panel {
            padding: 16px;
            overflow: auto;
        }
        
        .split-view .response-panel, .split-view .evidence-panel {
            width: 50%;
            height: 100%;
        }
        
        .stacked-view .response-panel, .stacked-view .evidence-panel {
            width: 100%;
            height: 50%;
        }
        
        .single-view .response-panel {
            width: 100%;
            height: 100%;
        }
        
        .full-width {
            width: 100%;
        }
        
        h3 {
            margin-top: 0;
            margin-bottom: 12px;
            color: var(--heading-color, #333);
            font-size: 16px;
        }
        
        .response-content {
            white-space: pre-wrap;
            font-family: var(--content-font, system-ui);
            line-height: 1.5;
        }
        
        .source-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            overflow-x: auto;
            padding-bottom: 4px;
        }
        
        .source-tab {
            background-color: var(--tab-bg, #e5e5e5);
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            white-space: nowrap;
        }
        
        .source-tab.active {
            background-color: var(--tab-active-bg, #3498db);
            color: var(--tab-active-color, white);
        }
        
        .source-tab:hover:not(.active) {
            background-color: var(--tab-hover-bg, #d5d5d5);
        }
        
        .no-evidence {
            padding: 24px;
            text-align: center;
            color: var(--muted-color, #888);
        }
        
        .loading {
            padding: 24px;
            text-align: center;
            color: var(--muted-color, #888);
        }
        
        .error {
            padding: 24px;
            text-align: center;
            color: var(--error-color, #e74c3c);
        }
        
        @keyframes flash {
            0% { background-color: rgba(255, 255, 0, 0.8); }
            100% { background-color: rgba(255, 204, 0, 0.3); }
        }
        
        .flash-highlight {
            animation: flash 1.5s ease-out;
        }
        
        .mention-source {
            border-left: 3px solid var(--mention-color, #6f42c1);
            background-color: var(--mention-background, rgba(111, 66, 193, 0.05));
        }
        
        .source-badge.mention {
            background-color: var(--mention-color, #6f42c1);
        }
        
        .source-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .source-tags {
            display: flex;
            gap: 4px;
        }
        
        .source-tag {
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: var(--tag-background, #f1f8ff);
            color: var(--tag-color, #0366d6);
        }
        
        .source-tag.mention {
            background-color: var(--mention-tag-background, #f5f0ff);
            color: var(--mention-color, #6f42c1);
        }
        
        .kg-source {
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            overflow: auto;
            height: 100%;
        }
        
        .kg-source h4 {
            margin-top: 0;
            color: #2a5885;
            border-bottom: 1px solid #e1e4e8;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        
        .kg-source h5 {
            margin: 15px 0 10px;
            color: #24292e;
            font-size: 1.1em;
        }
        
        .kg-entity-details {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 15px;
            background-color: #eef2f5;
            padding: 10px;
            border-radius: 6px;
        }
        
        .kg-entity-description {
            flex-basis: 100%;
            margin-top: 5px;
        }
        
        .properties-grid {
            display: grid;
            grid-template-columns: minmax(150px, 30%) 1fr;
            gap: 5px;
            background-color: white;
            border-radius: 6px;
            padding: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .property-row {
            display: contents;
        }
        
        .property-key {
            font-weight: 500;
            color: #586069;
            padding: 5px 10px;
            background-color: #f6f8fa;
            border-radius: 4px 0 0 4px;
        }
        
        .property-value {
            padding: 5px 10px;
            word-break: break-word;
            background-color: #fff;
            border-radius: 0 4px 4px 0;
            border-left: 3px solid #e1e4e8;
        }
        
        .relationships-list {
            background-color: white;
            border-radius: 6px;
            padding: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .relationship-item {
            display: flex;
            align-items: center;
            padding: 8px;
            margin: 5px 0;
            background-color: #f6f8fa;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .relationship-item:hover {
            background-color: #eef2f5;
        }
        
        .relationship-source, .relationship-target {
            flex: 1;
            padding: 0 5px;
        }
        
        .relationship-type {
            padding: 3px 8px;
            background-color: #e1e4e8;
            border-radius: 12px;
            font-size: 0.85em;
            color: #24292e;
            margin: 0 5px;
        }
        
        .relationship-arrow {
            color: #586069;
            margin: 0 5px;
        }
        
        .sources-list {
            list-style-type: none;
            padding: 0;
            margin: 0;
            background-color: white;
            border-radius: 6px;
            padding: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .source-item {
            padding: 8px;
            border-bottom: 1px solid #e1e4e8;
        }
        
        .source-item:last-child {
            border-bottom: none;
        }
        
        .source-date {
            font-size: 0.85em;
            color: #586069;
            margin-left: 10px;
        }
        
        .no-data {
            color: #586069;
            font-style: italic;
            padding: 10px;
            background-color: white;
            border-radius: 6px;
            text-align: center;
        }
        
        .kg-actions {
            margin-top: 20px;
            text-align: center;
        }
        
        .kg-explore-btn {
            background-color: #2a5885;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        .kg-explore-btn:hover {
            background-color: #1e3c5a;
        }
        
        .kg-explorer {
            margin-top: 20px;
            border-top: 1px solid #e1e4e8;
            padding-top: 15px;
        }
        
        .kg-graph-container {
            height: 400px;
            background-color: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .graph-placeholder {
            text-align: center;
            color: #586069;
            padding: 20px;
        }
    `;

    /**
     * Behandelt die Auswahl einer @-Mention und fügt sie als Quelle hinzu
     */
    async handleMentionSelected(event) {
        const mention = event.detail.mention;
        
        // Quelle nur hinzufügen, wenn sie noch nicht vorhanden ist
        if (!this.mentionSources.some(source => 
            source.type === mention.type && source.id === mention.id)) {
            
            this.loading = true;
            
            try {
                // Details für die Mention abrufen
                const response = await fetch(`/models/api/mentions/${mention.type}/${mention.id}`);
                if (response.ok) {
                    const data = await response.json();
                    
                    // Neue Mention-Quelle erstellen
                    const mentionSource = {
                        id: `mention-${mention.type}-${mention.id}`,
                        type: mention.type,
                        itemId: mention.id,
                        title: mention.text,
                        content: data.content || '',
                        highlights: [],
                        isMention: true,
                        metadata: data,
                        confidence: 1.0 // Explizite Erwähnung hat höchste Konfidenz
                    };
                    
                    // Zur Liste der Mention-Quellen hinzufügen
                    this.mentionSources = [...this.mentionSources, mentionSource];
                    
                    // Evidence Explorer öffnen, wenn es die erste Quelle ist
                    if (this.mentionSources.length === 1 && !this.isOpen) {
                        this.isOpen = true;
                    }
                }
            } catch (error) {
                console.error('Fehler beim Abrufen der Mention-Details:', error);
            } finally {
                this.loading = false;
            }
        }
    }

    /**
     * Behandelt Klicks auf Zitate im Chat
     */
    handleCitationClicked(event) {
        const citation = event.detail.citation;
        this.activeCitation = citation.id;
        
        // Quelle finden und aktivieren
        const source = this.findSourceForCitation(citation);
        if (source) {
            this.activeSource = source;
            this.isOpen = true;
            
            // Nach dem Rendern zum entsprechenden Highlight scrollen
            this.updateComplete.then(() => {
                this.scrollToHighlight(citation.highlightId);
            });
        }
    }

    /**
     * Findet die Quelle für ein bestimmtes Zitat
     */
    findSourceForCitation(citation) {
        // Zuerst in normalen Quellen suchen
        let source = this.sources.find(s => s.id === citation.sourceId);
        
        // Wenn nicht gefunden, in Mention-Quellen suchen
        if (!source) {
            source = this.mentionSources.find(s => s.id === citation.sourceId);
        }
        
        return source;
    }

    toggleKgExplorer() {
        this.showKgExplorer = !this.showKgExplorer;
        
        if (this.showKgExplorer) {
            // Wait for DOM to update, then initialize the graph
            this.updateComplete.then(() => {
                const source = this.evidenceSources[this.activeSourceIndex];
                if (source && source.entity) {
                    this.initializeKgGraph(source.entity);
                }
            });
        }
    }
    
    initializeKgGraph(entity) {
        // This is a placeholder for graph visualization
        // In a real implementation, you would use a library like D3.js, Cytoscape.js, or VisJS
        const container = this.shadowRoot.querySelector(`#kg-graph-${entity.id}`);
        if (!container) return;
        
        container.innerHTML = `
            <div class="graph-placeholder">
                <p>Graph visualization would be rendered here using a library like D3.js or Cytoscape.js</p>
                <p>Entity: ${entity.label} with ${entity.relationships?.length || 0} relationships</p>
            </div>
        `;
    }
    
    navigateToEntity(entityId) {
        // Find the entity in our sources
        const entityIndex = this.evidenceSources.findIndex(
            source => source.type === 'knowledge_graph' && source.entity?.id === entityId
        );
        
        if (entityIndex >= 0) {
            this.activeSourceIndex = entityIndex;
        } else {
            // Could fetch the entity from the API if not already loaded
            console.log(`Entity ${entityId} not loaded yet`);
        }
    }
}

customElements.define('evidence-explorer', EvidenceExplorer); 