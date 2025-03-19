import { html, css, LitElement } from 'lit';

class MetabaseEmbed extends LitElement {
    static properties = {
        dashboardId: { type: String },
        parameters: { type: Object },
        height: { type: String },
        loading: { type: Boolean },
        error: { type: String },
        token: { type: String }
    };

    constructor() {
        super();
        this.height = '600px';
        this.loading = true;
        this.error = null;
        this.parameters = {};
    }

    async firstUpdated() {
        try {
            await this.generateEmbedToken();
            this.loading = false;
        } catch (err) {
            this.error = 'Failed to load dashboard: ' + err.message;
            this.loading = false;
        }
    }

    async generateEmbedToken() {
        try {
            const response = await fetch('/api/metabase/embed-token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dashboard_id: this.dashboardId,
                    parameters: this.parameters
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.token = data.token;
        } catch (error) {
            console.error('Error generating embed token:', error);
            throw error;
        }
    }

    getIframeSrc() {
        if (!this.token) return '';
        
        const baseUrl = process.env.METABASE_URL || 'http://localhost:3000';
        return `${baseUrl}/embed/dashboard/${this.token}#bordered=true&titled=true`;
    }

    render() {
        if (this.loading) {
            return html`
                <div class="loading">
                    <div class="spinner"></div>
                    Loading dashboard...
                </div>
            `;
        }

        if (this.error) {
            return html`
                <div class="error">
                    <p>${this.error}</p>
                    <button @click="${this.retryLoad}">Retry</button>
                </div>
            `;
        }

        return html`
            <div class="metabase-container">
                <iframe
                    src="${this.getIframeSrc()}"
                    frameborder="0"
                    width="100%"
                    height="${this.height}"
                    allowtransparency
                    @load="${this.handleIframeLoad}"
                    @error="${this.handleIframeError}"
                ></iframe>

                <div class="controls">
                    <button @click="${this.refreshDashboard}" 
                            title="Refresh Dashboard">
                        🔄
                    </button>
                    <button @click="${this.exportData}"
                            title="Export Data">
                        📥
                    </button>
                </div>
            </div>
        `;
    }

    async refreshDashboard() {
        this.loading = true;
        await this.generateEmbedToken();
        this.loading = false;
    }

    async exportData() {
        // Implementation für Export-Funktionalität
        console.log('Exporting dashboard data...');
    }

    handleIframeLoad(e) {
        this.loading = false;
    }

    handleIframeError(e) {
        this.error = 'Failed to load dashboard';
    }

    async retryLoad() {
        this.error = null;
        this.loading = true;
        try {
            await this.generateEmbedToken();
        } catch (err) {
            this.error = 'Failed to reload dashboard: ' + err.message;
        }
        this.loading = false;
    }

    static styles = css`
        :host {
            display: block;
            position: relative;
        }

        .metabase-container {
            position: relative;
            width: 100%;
            height: 100%;
        }

        iframe {
            border: none;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .controls {
            position: absolute;
            top: 1rem;
            right: 1rem;
            display: flex;
            gap: 0.5rem;
            z-index: 100;
        }

        .controls button {
            padding: 0.5rem;
            border: none;
            border-radius: 4px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s;
        }

        .controls button:hover {
            transform: scale(1.1);
        }

        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            padding: 2rem;
            background: var(--loading-bg, #f5f5f5);
            border-radius: 8px;
        }

        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .error {
            padding: 2rem;
            text-align: center;
            color: var(--error-color, #dc3545);
            background: var(--error-bg, #fff);
            border-radius: 8px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
}

customElements.define('metabase-embed', MetabaseEmbed); 