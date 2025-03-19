import { html, css, LitElement } from 'lit';
import { repeat } from 'lit/directives/repeat.js';
import './performance-monitor.js';  // Importiere nur für Registrierung der Web Component

/**
 * Bruno API Testing Interface
 * 
 * Backend-Logik für API Testing, Monitoring und Debug-Features.
 * Diese Implementierung nutzt LitElement für Entwicklungs- und Test-Zwecke.
 * 
 * Für Production Frontend:
 * - Die Logik (detectBottlenecks, performanceMetrics etc.) kann direkt übernommen werden
 * - UI-Komponenten müssen im jeweiligen Frontend-Framework (React/Vue) neu implementiert werden
 * 
 * React-Beispiel:
 * ```jsx
 * const APIInterface = () => {
 *   const [endpoints, setEndpoints] = useState([]);
 *   const [performanceMetrics, setPerformanceMetrics] = useState({});
 *   
 *   const detectBottlenecks = () => {
 *     // gleiche Logik wie unten
 *   };
 *   
 *   return (
 *     <div className="api-container">
 *       // ... React-spezifische UI-Implementierung
 *     </div>
 *   );
 * };
 * ```
 */
class APIInterface extends LitElement {
    static properties = {
        endpoints: { type: Array },
        activeRequest: { type: Object },
        responseData: { type: Object },
        performanceMetrics: { type: Object },
        requestHistory: { type: Array },
        error: { type: String },
        showPerformanceMonitor: { type: Boolean }
    };

    constructor() {
        super();
        this.endpoints = [];
        this.activeRequest = null;
        this.responseData = null;
        this.performanceMetrics = {
            responseTime: 0,
            statusCode: null,
            dataSize: 0,
            memoryUsage: 0,
            dbQueryCount: 0,
            dbQueryTime: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
        this.requestHistory = [];
        this.error = null;
        this.showPerformanceMonitor = false;
    }

    static styles = css`
        :host {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .api-container {
            display: flex;
            height: 100%;
        }

        .endpoint-list {
            width: 250px;
            border-right: 1px solid var(--bruno-border-color, #e0e0e0);
            overflow-y: auto;
        }

        .request-viewer {
            flex: 1;
            padding: 1rem;
            display: flex;
            flex-direction: column;
        }

        .response-viewer {
            flex: 1;
            padding: 1rem;
            background: var(--bruno-response-bg, #f8f9fa);
            overflow-y: auto;
        }

        .metrics-panel {
            padding: 0.5rem;
            border-top: 1px solid var(--bruno-border-color, #e0e0e0);
            background: var(--bruno-metrics-bg, #fff);
        }

        .error {
            color: var(--bruno-error-color, #dc3545);
            padding: 0.5rem;
            margin: 0.5rem 0;
            border: 1px solid currentColor;
            border-radius: 4px;
        }

        .sql-queries {
            margin-top: 1rem;
            padding: 1rem;
            background: var(--bruno-code-bg, #f8f9fa);
            border-radius: 4px;
        }

        .query-list {
            max-height: 300px;
            overflow-y: auto;
        }

        .query-item {
            margin: 0.5rem 0;
            padding: 0.5rem;
            border: 1px solid var(--bruno-border-color, #e0e0e0);
            border-radius: 4px;
        }

        .query-item.slow-query {
            border-color: var(--bruno-warning-color, #ffc107);
        }

        .query-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }

        .query-time {
            color: var(--bruno-text-secondary, #6c757d);
        }

        .query-type {
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            background: var(--bruno-primary-color, #007bff);
            color: white;
        }

        .query-sql {
            margin: 0;
            padding: 0.5rem;
            background: var(--bruno-code-bg, #f8f9fa);
            border-radius: 3px;
            font-family: monospace;
            white-space: pre-wrap;
        }

        .query-warning {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: var(--bruno-warning-bg, #fff3cd);
            color: var(--bruno-warning-text, #856404);
            border-radius: 3px;
        }

        .cache-status {
            margin-top: 1rem;
            padding: 1rem;
            background: var(--bruno-bg-secondary, #f8f9fa);
            border-radius: 4px;
        }

        .cache-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .cache-metric {
            padding: 0.5rem;
            background: white;
            border-radius: 4px;
            text-align: center;
        }

        .cache-metric span {
            display: block;
            color: var(--bruno-text-secondary, #6c757d);
            font-size: 0.9em;
        }

        .cache-metric strong {
            display: block;
            font-size: 1.2em;
            margin-top: 0.2rem;
        }

        .cache-entries {
            max-height: 200px;
            overflow-y: auto;
        }

        .cache-entry {
            margin: 0.5rem 0;
            padding: 0.5rem;
            background: white;
            border: 1px solid var(--bruno-border-color, #e0e0e0);
            border-radius: 4px;
        }

        .cache-entry.expired {
            border-color: var(--bruno-warning-color, #ffc107);
            background: var(--bruno-warning-bg, #fff3cd);
        }

        .entry-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.3rem;
        }

        .entry-key {
            font-family: monospace;
            font-size: 0.9em;
        }

        .entry-age {
            color: var(--bruno-text-secondary, #6c757d);
            font-size: 0.9em;
        }

        .entry-info {
            display: flex;
            gap: 1rem;
            font-size: 0.9em;
            color: var(--bruno-text-secondary, #6c757d);
        }

        .history-panel {
            margin-top: 1rem;
            padding: 1rem;
            background: var(--bruno-bg-secondary, #f8f9fa);
            border-radius: 4px;
            max-height: 300px;
            overflow-y: auto;
        }

        .history-item {
            margin: 0.5rem 0;
            padding: 0.5rem;
            background: white;
            border: 1px solid var(--bruno-border-color, #e0e0e0);
            border-radius: 4px;
        }

        .history-item.has-bottlenecks {
            border-color: var(--bruno-warning-color, #ffc107);
        }

        .history-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
        }

        .history-metrics {
            display: flex;
            gap: 1rem;
            margin-top: 0.3rem;
            font-size: 0.8em;
            color: var(--bruno-text-secondary, #6c757d);
        }

        .history-bottlenecks {
            margin-top: 0.3rem;
            font-size: 0.8em;
            color: var(--bruno-warning-color, #ffc107);
        }

        .database-status {
            margin-top: 1rem;
            padding: 1rem;
            background: var(--bruno-bg-secondary, #f8f9fa);
            border-radius: 4px;
        }

        .db-section {
            margin-top: 1rem;
        }

        .db-section h5 {
            margin: 0 0 0.5rem 0;
            color: var(--bruno-text-secondary, #6c757d);
        }

        .table-item, .index-item {
            margin: 0.5rem 0;
            padding: 0.5rem;
            background: white;
            border: 1px solid var(--bruno-border-color, #e0e0e0);
            border-radius: 4px;
        }

        .table-header, .index-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.3rem;
        }

        .table-metrics, .index-metrics {
            display: flex;
            gap: 1rem;
            font-size: 0.9em;
            color: var(--bruno-text-secondary, #6c757d);
        }

        .index-item.low-usage {
            border-color: var(--bruno-warning-color, #ffc107);
            background: var(--bruno-warning-bg, #fff3cd);
        }

        .index-warning {
            margin-top: 0.5rem;
            color: var(--bruno-warning-text, #856404);
            font-size: 0.9em;
        }

        .pool-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        .pool-metric {
            padding: 0.5rem;
            background: white;
            border-radius: 4px;
            text-align: center;
        }

        .pool-metric span {
            display: block;
            color: var(--bruno-text-secondary, #6c757d);
            font-size: 0.9em;
        }

        .pool-metric strong {
            display: block;
            font-size: 1.2em;
            margin-top: 0.2rem;
        }

        .export-buttons {
            margin-top: 1rem;
            display: flex;
            gap: 0.5rem;
        }

        .export-buttons button {
            padding: 0.5rem 1rem;
            border: 1px solid var(--bruno-border-color, #e0e0e0);
            border-radius: 4px;
            background: white;
            cursor: pointer;
        }

        .export-buttons button:hover {
            background: var(--bruno-hover-bg, #f5f5f5);
        }

        .performance-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }

        .performance-overlay {
            position: fixed;
            bottom: 60px;
            right: 20px;
            width: 400px;
            background: var(--surface-color, #ffffff);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 999;
        }
    `;

    render() {
        return html`
            ${this.renderMainInterface()}
            ${this.renderPerformanceMonitor()}
        `;
    }

    renderMainInterface() {
        return html`
            <div class="api-container">
                <div class="endpoint-list">
                    ${this.renderEndpoints()}
                </div>
                <div class="request-viewer">
                    ${this.renderActiveRequest()}
                    <div class="response-viewer">
                        ${this.renderResponse()}
                    </div>
                    <div class="metrics-panel">
                        ${this.renderMetrics()}
                        ${this.renderBottlenecks()}
                        ${this.renderExportButtons()}
                    </div>
                </div>
            </div>
        `;
    }

    renderEndpoints() {
        return html`
            <h3>API Endpoints</h3>
            ${repeat(
                this.endpoints,
                endpoint => endpoint.id,
                endpoint => html`
                    <div class="endpoint-item" @click=${() => this.selectEndpoint(endpoint)}>
                        <span>${endpoint.method}</span>
                        <span>${endpoint.path}</span>
                    </div>
                `
            )}
        `;
    }

    renderActiveRequest() {
        if (!this.activeRequest) return html`<p>Select an endpoint to test</p>`;

        return html`
            <div class="request-panel">
                <h3>Request</h3>
                <div class="request-method">
                    ${this.activeRequest.method} ${this.activeRequest.path}
                </div>
                <div class="request-headers">
                    <h4>Headers</h4>
                    ${this.renderHeaders(this.activeRequest.headers)}
                </div>
                <div class="request-body">
                    <h4>Body</h4>
                    <textarea
                        .value=${JSON.stringify(this.activeRequest.body, null, 2)}
                        @input=${this.handleRequestBodyChange}
                    ></textarea>
                </div>
                <button @click=${this.sendRequest}>Send Request</button>
            </div>
        `;
    }

    renderResponse() {
        if (!this.responseData) return html`<p>No response data</p>`;

        return html`
            <h3>Response</h3>
            <div class="response-status">
                Status: ${this.responseData.status} ${this.responseData.statusText}
            </div>
            <div class="response-headers">
                <h4>Headers</h4>
                ${this.renderHeaders(this.responseData.headers)}
            </div>
            <div class="response-body">
                <h4>Body</h4>
                <pre>${JSON.stringify(this.responseData.body, null, 2)}</pre>
            </div>
        `;
    }

    renderMetrics() {
        return html`
            <div class="metrics-grid">
                ${this.renderBasicMetrics()}
            </div>
            ${this.renderSQLQueries()}
            ${this.renderCacheStatus()}
            ${this.renderDatabaseStatus()}
        `;
    }

    renderBasicMetrics() {
        return html`
            <div class="metrics-grid">
                <div class="metric">
                    <span>Response Time</span>
                    <strong>${this.performanceMetrics.responseTime}ms</strong>
                </div>
                <div class="metric">
                    <span>Status Code</span>
                    <strong>${this.performanceMetrics.statusCode}</strong>
                </div>
                <div class="metric">
                    <span>Data Size</span>
                    <strong>${this.formatBytes(this.performanceMetrics.dataSize)}</strong>
                </div>
                <div class="metric">
                    <span>DB Queries</span>
                    <strong>${this.performanceMetrics.dbQueryCount}</strong>
                </div>
                <div class="metric">
                    <span>Query Time</span>
                    <strong>${this.performanceMetrics.dbQueryTime}ms</strong>
                </div>
                <div class="metric">
                    <span>Cache Hit Rate</span>
                    <strong>${this.calculateCacheHitRate()}%</strong>
                </div>
            </div>
        `;
    }

    renderSQLQueries() {
        if (!this.responseData?.debug?.sql_queries) return '';

        return html`
            <div class="sql-queries">
                <h4>SQL Queries</h4>
                <div class="query-list">
                    ${repeat(
                        this.responseData.debug.sql_queries,
                        (query, index) => index,
                        query => html`
                            <div class="query-item ${query.duration > 100 ? 'slow-query' : ''}">
                                <div class="query-header">
                                    <span class="query-time">${query.duration}ms</span>
                                    <span class="query-type">${this.getQueryType(query.sql)}</span>
                                </div>
                                <pre class="query-sql">${query.sql}</pre>
                                ${query.duration > 100 ? html`
                                    <div class="query-warning">
                                        Slow Query Warning: Consider optimization
                                    </div>
                                ` : ''}
                            </div>
                        `
                    )}
                </div>
            </div>
        `;
    }

    renderCacheStatus() {
        if (!this.responseData?.debug?.cache_stats) return '';

        return html`
            <div class="cache-status">
                <h4>Cache Status</h4>
                <div class="cache-grid">
                    <div class="cache-metric">
                        <span>Hits</span>
                        <strong>${this.performanceMetrics.cacheHits}</strong>
                    </div>
                    <div class="cache-metric">
                        <span>Misses</span>
                        <strong>${this.performanceMetrics.cacheMisses}</strong>
                    </div>
                    <div class="cache-metric">
                        <span>Memory Usage</span>
                        <strong>${this.formatBytes(this.performanceMetrics.memoryUsage)}</strong>
                    </div>
                </div>
                <div class="cache-entries">
                    ${repeat(
                        this.responseData.debug.cache_stats.entries || [],
                        entry => entry.key,
                        entry => html`
                            <div class="cache-entry ${entry.expired ? 'expired' : ''}">
                                <div class="entry-header">
                                    <span class="entry-key">${entry.key}</span>
                                    <span class="entry-age">${this.formatAge(entry.age)}</span>
                                </div>
                                <div class="entry-info">
                                    <span>Size: ${this.formatBytes(entry.size)}</span>
                                    <span>TTL: ${entry.ttl}s</span>
                                </div>
                            </div>
                        `
                    )}
                </div>
            </div>
        `;
    }

    renderDatabaseStatus() {
        if (!this.responseData?.debug?.db_stats) return '';

        return html`
            <div class="database-status">
                <h4>Database Status</h4>
                
                <!-- Tabellen Status -->
                <div class="db-section">
                    <h5>Tables</h5>
                    <div class="table-stats">
                        ${repeat(
                            this.responseData.debug.db_stats.tables || [],
                            table => table.name,
                            table => html`
                                <div class="table-item">
                                    <div class="table-header">
                                        <span class="table-name">${table.name}</span>
                                        <span class="table-size">${this.formatBytes(table.size)}</span>
                                    </div>
                                    <div class="table-metrics">
                                        <span>Rows: ${table.row_count}</span>
                                        <span>Indexes: ${table.index_count}</span>
                                        <span>Last Vacuum: ${this.formatTimestamp(table.last_vacuum)}</span>
                                    </div>
                                </div>
                            `
                        )}
                    </div>
                </div>

                <!-- Index Nutzung -->
                <div class="db-section">
                    <h5>Index Usage</h5>
                    <div class="index-stats">
                        ${repeat(
                            this.responseData.debug.db_stats.indexes || [],
                            index => index.name,
                            index => html`
                                <div class="index-item ${index.usage_ratio < 0.1 ? 'low-usage' : ''}">
                                    <div class="index-header">
                                        <span class="index-name">${index.name}</span>
                                        <span class="index-table">${index.table}</span>
                                    </div>
                                    <div class="index-metrics">
                                        <span>Scans: ${index.scans}</span>
                                        <span>Size: ${this.formatBytes(index.size)}</span>
                                        <span>Usage: ${(index.usage_ratio * 100).toFixed(1)}%</span>
                                    </div>
                                    ${index.usage_ratio < 0.1 ? html`
                                        <div class="index-warning">
                                            Low usage index - Consider removing
                                        </div>
                                    ` : ''}
                                </div>
                            `
                        )}
                    </div>
                </div>

                <!-- Connection Pool -->
                <div class="db-section">
                    <h5>Connection Pool</h5>
                    <div class="pool-stats">
                        <div class="pool-metric">
                            <span>Active</span>
                            <strong>${this.responseData.debug.db_stats.pool?.active || 0}</strong>
                        </div>
                        <div class="pool-metric">
                            <span>Idle</span>
                            <strong>${this.responseData.debug.db_stats.pool?.idle || 0}</strong>
                        </div>
                        <div class="pool-metric">
                            <span>Max</span>
                            <strong>${this.responseData.debug.db_stats.pool?.max || 0}</strong>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getQueryType(sql) {
        sql = sql.trim().toUpperCase();
        if (sql.startsWith('SELECT')) return 'SELECT';
        if (sql.startsWith('INSERT')) return 'INSERT';
        if (sql.startsWith('UPDATE')) return 'UPDATE';
        if (sql.startsWith('DELETE')) return 'DELETE';
        return 'OTHER';
    }

    // Utility-Methoden
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    calculateCacheHitRate() {
        const total = this.performanceMetrics.cacheHits + this.performanceMetrics.cacheMisses;
        if (total === 0) return 0;
        return ((this.performanceMetrics.cacheHits / total) * 100).toFixed(1);
    }

    formatAge(seconds) {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        return `${Math.floor(seconds / 3600)}h`;
    }

    // Event Handler
    async sendRequest() {
        if (!this.activeRequest) return;

        const startTime = performance.now();
        try {
            const response = await fetch(this.activeRequest.path, {
                method: this.activeRequest.method,
                headers: this.activeRequest.headers,
                body: JSON.stringify(this.activeRequest.body)
            });

            const responseData = await response.json();
            const endTime = performance.now();

            this.responseData = {
                status: response.status,
                statusText: response.statusText,
                headers: Object.fromEntries(response.headers),
                body: responseData,
                debug: responseData.debug || {}
            };

            this.performanceMetrics = {
                responseTime: Math.round(endTime - startTime),
                statusCode: response.status,
                dataSize: new Blob([JSON.stringify(responseData)]).size,
                dbQueryCount: this.responseData.debug?.sql_queries?.length || 0,
                dbQueryTime: this.responseData.debug?.sql_queries?.reduce((sum, q) => sum + q.duration, 0) || 0,
                cacheHits: this.responseData.debug?.cache_stats?.hits || 0,
                cacheMisses: this.responseData.debug?.cache_stats?.misses || 0,
                memoryUsage: this.responseData.debug?.cache_stats?.memory_usage || 0
            };

            // Bottleneck Detection nach jedem Request
            const bottlenecks = this.detectBottlenecks();
            if (bottlenecks.length > 0) {
                console.warn('Performance Bottlenecks detected:', bottlenecks);
            }

            this.requestHistory = [...this.requestHistory, {
                request: this.activeRequest,
                response: this.responseData,
                metrics: this.performanceMetrics,
                bottlenecks: bottlenecks,
                timestamp: new Date()
            }];

        } catch (error) {
            this.error = error.message;
            console.error('API Request Error:', error);
        }
    }

    renderRequestHistory() {
        return html`
            <div class="history-panel">
                <h4>Request History</h4>
                ${repeat(
                    this.requestHistory,
                    (entry) => entry.timestamp,
                    (entry) => html`
                        <div class="history-item ${entry.bottlenecks.length > 0 ? 'has-bottlenecks' : ''}">
                            <div class="history-header">
                                <span>${entry.request.method} ${entry.request.path}</span>
                                <span>${this.formatTimestamp(entry.timestamp)}</span>
                            </div>
                            <div class="history-metrics">
                                <span>${entry.metrics.responseTime}ms</span>
                                <span>${entry.metrics.statusCode}</span>
                                <span>${this.formatBytes(entry.metrics.dataSize)}</span>
                            </div>
                            ${entry.bottlenecks.length > 0 ? html`
                                <div class="history-bottlenecks">
                                    <span>⚠️ ${entry.bottlenecks.length} bottlenecks detected</span>
                                </div>
                            ` : ''}
                        </div>
                    `
                )}
            </div>
        `;
    }

    /**
     * Exportiert die Debug-Daten in verschiedenen Formaten
     */
    exportData(format = 'json') {
        const exportData = {
            timestamp: new Date().toISOString(),
            endpoint: this.activeRequest?.path,
            request: {
                method: this.activeRequest?.method,
                headers: this.activeRequest?.headers,
                body: this.activeRequest?.body
            },
            response: this.responseData,
            performance: this.performanceMetrics,
            sqlQueries: this.responseData?.debug?.sql_queries || [],
            cacheStats: this.responseData?.debug?.cache_stats || {},
            dbStats: this.responseData?.debug?.db_stats || {},
            bottlenecks: this.detectBottlenecks()
        };

        switch (format) {
            case 'json':
                this.downloadFile(
                    JSON.stringify(exportData, null, 2),
                    'api-debug.json',
                    'application/json'
                );
                break;
            case 'html':
                this.generateHTMLReport(exportData);
                break;
            case 'csv':
                this.exportMetricsCSV(exportData);
                break;
        }
    }

    /**
     * Generiert einen HTML-Report
     */
    generateHTMLReport(data) {
        const html = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>API Debug Report</title>
                <style>
                    body { font-family: sans-serif; padding: 20px; }
                    .section { margin: 20px 0; }
                    .metric { margin: 10px 0; }
                    pre { background: #f5f5f5; padding: 10px; }
                </style>
            </head>
            <body>
                <h1>API Debug Report</h1>
                <div class="section">
                    <h2>Request Details</h2>
                    <div class="metric">Endpoint: ${data.endpoint}</div>
                    <div class="metric">Method: ${data.request.method}</div>
                    <pre>${JSON.stringify(data.request.body, null, 2)}</pre>
                </div>
                <div class="section">
                    <h2>Performance Metrics</h2>
                    <div class="metric">Response Time: ${data.performance.responseTime}ms</div>
                    <div class="metric">DB Queries: ${data.performance.dbQueryCount}</div>
                    <div class="metric">Cache Hit Rate: ${this.calculateCacheHitRate()}%</div>
                </div>
                <!-- ... weitere Sections ... -->
            </body>
            </html>
        `;
        this.downloadFile(html, 'api-report.html', 'text/html');
    }

    /**
     * Exportiert Performance-Metriken als CSV
     */
    exportMetricsCSV(data) {
        const headers = ['Timestamp', 'Endpoint', 'Method', 'Response Time', 'Status', 'DB Queries', 'Cache Hits', 'Cache Misses'];
        const rows = [[
            data.timestamp,
            data.endpoint,
            data.request.method,
            data.performance.responseTime,
            data.response.status,
            data.performance.dbQueryCount,
            data.performance.cacheHits,
            data.performance.cacheMisses
        ]];

        const csv = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        this.downloadFile(csv, 'api-metrics.csv', 'text/csv');
    }

    /**
     * Hilfsfunktion zum Herunterladen von Dateien
     */
    downloadFile(content, filename, type) {
        const blob = new Blob([content], { type });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    // Füge Export-Button zum UI hinzu
    renderExportButtons() {
        return html`
            <div class="export-buttons">
                <button @click=${() => this.exportData('json')}>Export JSON</button>
                <button @click=${() => this.exportData('html')}>Generate Report</button>
                <button @click=${() => this.exportData('csv')}>Export Metrics CSV</button>
            </div>
        `;
    }

    renderPerformanceMonitor() {
        return html`
            <button 
                class="performance-toggle"
                @click="${() => this.showPerformanceMonitor = !this.showPerformanceMonitor}">
                ${this.showPerformanceMonitor ? 'Hide' : 'Show'} Performance
            </button>

            ${this.showPerformanceMonitor ? html`
                <div class="performance-overlay">
                    <bruno-performance-monitor
                        .metrics="${this.performanceMetrics}"
                        .historicalData="${this.requestHistory}"
                    ></bruno-performance-monitor>
                </div>
            ` : ''}
        `;
    }
}

customElements.define('bruno-api-interface', APIInterface);

export default APIInterface;
