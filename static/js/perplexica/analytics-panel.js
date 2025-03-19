import { html, css, LitElement } from 'lit';
import './metabase-embed.js';
import { Chart } from 'chart.js/auto';

class AnalyticsPanel extends LitElement {
    static properties = {
        data: { type: Object },
        activeView: { type: String },
        timeRange: { type: String },
        loading: { type: Boolean },
        error: { type: String },
        dashboards: { type: Array }
    };

    constructor() {
        super();
        this.data = {};
        this.activeView = 'overview';
        this.timeRange = 'last_7_days';
        this.loading = false;
        this.error = null;
        this.dashboards = [];
        this.charts = new Map();
    }

    firstUpdated() {
        this.initializeCharts();
    }

    initializeCharts() {
        // Overview Charts
        const ctx = this.shadowRoot.querySelector('#activity-chart').getContext('2d');
        this.charts.set('activity', new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'User Activity',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        }));
    }

    updateCharts() {
        const activityChart = this.charts.get('activity');
        if (activityChart && this.data.activity) {
            activityChart.data.labels = this.data.activity.map(d => d.date);
            activityChart.data.datasets[0].data = this.data.activity.map(d => d.value);
            activityChart.update();
        }
    }

    render() {
        return html`
            <div class="analytics-panel">
                <!-- Header -->
                <div class="header">
                    <h2>Analytics Dashboard</h2>
                    <div class="controls">
                        <select .value="${this.timeRange}" 
                                @change="${e => this.timeRange = e.target.value}">
                            <option value="today">Today</option>
                            <option value="last_7_days">Last 7 Days</option>
                            <option value="last_30_days">Last 30 Days</option>
                            <option value="last_90_days">Last 90 Days</option>
                        </select>
                    </div>
                </div>

                <!-- Navigation -->
                <div class="nav-tabs">
                    <button class="${this.activeView === 'overview' ? 'active' : ''}"
                            @click="${() => this.activeView = 'overview'}">
                        Overview
                    </button>
                    <button class="${this.activeView === 'documents' ? 'active' : ''}"
                            @click="${() => this.activeView = 'documents'}">
                        Documents
                    </button>
                    <button class="${this.activeView === 'search' ? 'active' : ''}"
                            @click="${() => this.activeView = 'search'}">
                        Search Analytics
                    </button>
                    <button class="${this.activeView === 'metabase' ? 'active' : ''}"
                            @click="${() => this.activeView = 'metabase'}">
                        Metabase
                    </button>
                </div>

                <!-- Content -->
                <div class="content">
                    ${this.loading ? 
                        html`<div class="loading">Loading analytics...</div>` :
                        this.renderContent()}
                </div>
            </div>
        `;
    }

    renderContent() {
        if (this.error) {
            return html`<div class="error">${this.error}</div>`;
        }

        switch (this.activeView) {
            case 'overview':
                return this.renderOverview();
            case 'documents':
                return this.renderDocuments();
            case 'search':
                return this.renderSearch();
            case 'metabase':
                return this.renderMetabase();
            default:
                return html`<div>Select a view</div>`;
        }
    }

    renderOverview() {
        return html`
            <div class="overview">
                <!-- KPI Cards -->
                <div class="kpi-grid">
                    ${this.renderKPICard('Total Users', this.data.totalUsers || 0)}
                    ${this.renderKPICard('Active Documents', this.data.activeDocuments || 0)}
                    ${this.renderKPICard('Search Queries', this.data.searchQueries || 0)}
                    ${this.renderKPICard('Success Rate', `${this.data.successRate || 0}%`)}
                </div>

                <!-- Activity Chart -->
                <div class="chart-container">
                    <canvas id="activity-chart"></canvas>
                </div>
            </div>
        `;
    }

    renderDocuments() {
        return html`
            <div class="documents">
                <metabase-embed
                    .dashboardId="${'document_analytics'}"
                    .parameters="${{timeRange: this.timeRange}}"
                ></metabase-embed>
            </div>
        `;
    }

    renderSearch() {
        return html`
            <div class="search-analytics">
                <metabase-embed
                    .dashboardId="${'search_analytics'}"
                    .parameters="${{timeRange: this.timeRange}}"
                ></metabase-embed>
            </div>
        `;
    }

    renderMetabase() {
        return html`
            <div class="metabase-dashboards">
                ${this.dashboards.map(dashboard => html`
                    <metabase-embed
                        .dashboardId="${dashboard.id}"
                        .parameters="${dashboard.parameters}"
                    ></metabase-embed>
                `)}
            </div>
        `;
    }

    renderKPICard(title, value) {
        return html`
            <div class="kpi-card">
                <h3>${title}</h3>
                <div class="value">${value}</div>
            </div>
        `;
    }

    static styles = css`
        :host {
            display: block;
            height: 100%;
        }

        .analytics-panel {
            display: flex;
            flex-direction: column;
            height: 100%;
            background: var(--panel-bg, #fff);
            border-radius: 8px;
            overflow: hidden;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background: var(--header-bg, #f5f5f5);
        }

        .nav-tabs {
            display: flex;
            gap: 0.5rem;
            padding: 0.5rem;
            background: var(--tabs-bg, #fff);
            border-bottom: 1px solid var(--border-color, #ddd);
        }

        .nav-tabs button {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            background: none;
            cursor: pointer;
        }

        .nav-tabs button.active {
            background: var(--primary-color, #007bff);
            color: white;
        }

        .content {
            flex: 1;
            overflow: auto;
            padding: 1rem;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .kpi-card {
            padding: 1rem;
            background: var(--card-bg, #f8f9fa);
            border-radius: 4px;
            text-align: center;
        }

        .chart-container {
            height: 300px;
            margin-top: 1rem;
        }

        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }

        .error {
            color: var(--error-color, #dc3545);
            padding: 1rem;
            text-align: center;
        }
    `;
}

customElements.define('analytics-panel', AnalyticsPanel); 