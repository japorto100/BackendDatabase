import { LitElement, html, css } from 'lit';
import { repeat } from 'lit/directives/repeat.js';
import {
    Chart,
    LineController,
    BarController,
    PieController,
    ScatterController,
    CategoryScale,
    LinearScale,
    TimeScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Tooltip,
    Legend
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import Chart from 'chart.js/auto';
import { AlertTemplates, AlertTemplateManager } from './alert-templates.js';
import { ExportManager } from './export-manager.js';
// Worker Import
import Worker from './performance-worker.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import annotationPlugin from 'chartjs-plugin-annotation';
import { format } from 'date-fns';
import { LitVirtualizer } from '@lit-labs/virtualizer';
import * as d3 from 'd3';

// Registriere Chart.js Plugins
Chart.register(zoomPlugin);
Chart.register(annotationPlugin);

/**
 * Performance Monitor Component
 * 
 * Zeigt Performance-Metriken in Echtzeit an und ermöglicht detaillierte Analysen
 */
class PerformanceMonitor extends LitElement {
    static properties = {
        metrics: { type: Object },
        timeRange: { type: String },
        selectedMetric: { type: String },
        historicalData: { type: Array },
        isExpanded: { type: Boolean },
        alertThresholds: { type: Object },
        charts: { state: true },
        compareMode: { type: Boolean },
        comparedMetrics: { type: Array },
        showStats: { type: Boolean },
        exportFormat: { type: String },
        alerts: { type: Array },
        thresholds: { type: Object },
        notificationsEnabled: { type: Boolean },
        alertHistory: { type: Array },
        chartType: { type: String },
        visualizationMode: { type: String },
        customRanges: { type: Array },
        worker: { type: Worker },
        _processSynchronously: { type: Boolean },
        dataResolution: { type: String },
        alertTemplateManager: { state: true },
        activeTemplates: { type: Array },
        selectedTemplate: { type: String },
        exportManager: { state: true },
        exportInProgress: { type: Boolean },
        exportError: { type: String },
        zoomMode: { type: String },
        annotations: { type: Array },
        selectedDataPoint: { type: Object },
        comparisonMode: { type: Boolean },
        comparisonPoints: { type: Array },
        visibleTimeRange: { type: Object },
        dataChunkSize: { type: Number },
        activeChunks: { type: Array },
        activeVisualization: { type: String },
        heatmapData: { type: Array },
        waterfallData: { type: Array },
        stackTraceData: { type: Array },
        flameData: { type: Array },
        totalTime: { type: Number },
        correlationData: { type: Array },
        xAxisMetric: { type: String },
        yAxisMetric: { type: String }
    };

    constructor() {
        super();
        this.metrics = {
            avgResponseTime: 0,
            messageCount: 0,
            tokenUsage: 0,
            cacheHitRate: 0,
            dbQueryCount: 0,
            memoryUsage: 0,
            activeConnections: 0,
            wsLatency: 0,
            apiCallsPerMinute: 0,
            errorRate: 0
        };
        this.timeRange = '1h'; // 1h, 24h, 7d
        this.selectedMetric = 'avgResponseTime';
        this.historicalData = [];
        this.isExpanded = false;
        this.alertThresholds = {
            avgResponseTime: 1000, // ms
            errorRate: 5, // %
            memoryUsage: 80, // %
            cacheHitRate: 50 // %
        };
        this.charts = {};
        this.compareMode = false;
        this.comparedMetrics = [];
        this.showStats = false;
        this.exportFormat = 'json';
        this.alerts = [];
        this.alertHistory = [];
        this.notificationsEnabled = false;
        this.thresholds = {
            avgResponseTime: { warning: 500, critical: 1000 },
            errorRate: { warning: 1, critical: 5 },
            memoryUsage: { warning: 70, critical: 85 },
            cacheHitRate: { warning: 60, critical: 40 },
            dbQueryCount: { warning: 50, critical: 100 },
            wsLatency: { warning: 100, critical: 200 }
        };
        this.chartType = 'line';
        this.visualizationMode = 'standard';
        this.customRanges = [
            { start: 0, end: 100, color: '#4caf50' },     // Gut
            { start: 100, end: 500, color: '#ffc107' },   // Warnung
            { start: 500, end: Infinity, color: '#f44336' } // Kritisch
        ];
        this._processSynchronously = false;
        this.dataResolution = '1h'; // 1h, 24h, 7d
        this.alertTemplateManager = new AlertTemplateManager();
        this.activeTemplates = [];
        this.selectedTemplate = '';
        this.exportManager = new ExportManager();
        this.exportInProgress = false;
        this.exportError = '';
        this.zoomMode = 'x';  // 'x', 'y', or 'xy'
        this.annotations = [];
        this.selectedDataPoint = null;
        this.comparisonMode = false;
        this.comparisonPoints = [];
        this.dataChunkSize = 1000; // Datenpunkte pro Chunk
        this.activeChunks = [];
        this.visibleTimeRange = {
            start: Date.now() - 24 * 60 * 60 * 1000, // 24h
            end: Date.now()
        };
        this.activeVisualization = 'line'; // 'line', 'heatmap', 'waterfall', 'stacktrace'
        this.heatmapData = [];
        this.waterfallData = [];
        this.stackTraceData = [];
        this.flameData = [];
        this.totalTime = 0;
        this.correlationData = [];
        this.xAxisMetric = 'Avg Response Time';
        this.yAxisMetric = 'Message Count';
        
        // Memory Management
        this._initMemoryManager();

        // Start metrics collection
        this.startMetricsCollection();
        this._startAlertMonitoring();
        this._requestNotificationPermission();
        
        // Worker Initialisierung
        this._initWorker();

        // Chunk Management
        this._initChunkManagement();

        // Chart Synchronisation
        this._initChartSync();
    }

    static get styles() {
        return css`
            ${super.styles}
            
            /* Layout Container */
            .performance-monitor {
                padding: 20px;
                background: var(--background-color, #f8f9fa);
                border-radius: 8px;
            }

            /* Visualization Controls */
            .visualization-controls {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                padding: 10px;
                background: var(--surface-color, #fff);
                border-radius: 6px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .control-button {
                padding: 8px 16px;
                border: 1px solid var(--border-color, #ddd);
                border-radius: 4px;
                background: var(--surface-color, #fff);
                color: var(--text-secondary-color, #666);
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .control-button.active {
                background: var(--primary-color, #2196f3);
                color: white;
                border-color: var(--primary-color, #2196f3);
            }

            .control-button:hover:not(.active) {
                background: var(--hover-color, #f5f5f5);
            }

            /* View Container */
            .view {
                display: block;
                opacity: 1;
                transition: opacity 0.3s ease;
            }

            .view.hidden {
                display: none;
                opacity: 0;
            }

            /* Metric Selectors */
            .metric-selectors {
                margin-bottom: 20px;
                padding: 15px;
                background: var(--surface-secondary-color, #f5f5f5);
                border-radius: 4px;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .metric-selectors select {
                padding: 8px;
                border: 1px solid var(--border-color, #ddd);
                border-radius: 4px;
                background: white;
                font-size: 14px;
                color: var(--text-primary-color, #333);
            }

            /* Loading States */
            .loading {
                position: relative;
            }

            .loading::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(255, 255, 255, 0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 14px;
                color: var(--text-secondary-color, #666);
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .visualization-controls {
                    flex-wrap: wrap;
                }

                .control-button {
                    flex: 1 1 calc(50% - 10px);
                }

                .metric-selectors {
                    flex-direction: column;
                    align-items: stretch;
                }
            }

            /* Dark Mode Support */
            @media (prefers-color-scheme: dark) {
                .performance-monitor {
                    background: var(--dark-background, #1a1a1a);
                }

                .control-button:not(.active) {
                    background: var(--dark-surface, #2d2d2d);
                    color: var(--dark-text, #e0e0e0);
                    border-color: var(--dark-border, #404040);
                }

                .metric-selectors {
                    background: var(--dark-surface-secondary, #2d2d2d);
                }

                .loading::after {
                    background: rgba(0, 0, 0, 0.8);
                    color: var(--dark-text, #e0e0e0);
                }
            }

            /* Flame Graph Styles */
            .flame-graph {
                width: 100%;
                height: 400px;
                background: var(--surface-color, #fff);
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .flame-tooltip {
                position: absolute;
                background: rgba(0, 0, 0, 0.85);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
            }

            /* Correlation Plot Styles */
            .correlation-plot {
                width: 100%;
                height: 400px;
                padding: 20px;
                background: var(--surface-color, #fff);
                border-radius: 4px;
            }

            .correlation-tooltip {
                position: absolute;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid var(--border-color, #ddd);
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                z-index: 1000;
            }

            .axis-label {
                font-size: 12px;
                fill: var(--text-secondary-color, #666);
            }

            .trend-line {
                stroke: var(--primary-color, #2196f3);
                stroke-width: 2;
                stroke-dasharray: 4;
            }

            /* System Resource Map Styles */
            .resource-map {
                width: 100%;
                height: 500px;
                background: var(--surface-color, #fff);
                border-radius: 4px;
                padding: 20px;
            }

            .resource-cell {
                transition: opacity 0.2s;
                cursor: pointer;
            }

            .resource-cell:hover {
                opacity: 0.8;
            }

            .resource-tooltip {
                position: absolute;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid var(--border-color, #ddd);
                border-radius: 4px;
                padding: 12px;
                font-size: 12px;
                pointer-events: none;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                z-index: 1000;
            }

            .tooltip-header {
                font-weight: 500;
                margin-bottom: 8px;
                padding-bottom: 4px;
                border-bottom: 1px solid var(--border-color, #ddd);
            }

            .tooltip-content {
                color: var(--text-secondary-color, #666);
                line-height: 1.4;
            }

            .legend {
                font-size: 10px;
            }

            .legend-label {
                fill: var(--text-secondary-color, #666);
            }

            /* Gemeinsame Styles */
            .visualization-container {
                margin: 20px 0;
                border: 1px solid var(--border-color, #ddd);
                border-radius: 4px;
                overflow: hidden;
            }

            .visualization-header {
                padding: 12px 16px;
                background: var(--surface-secondary-color, #f5f5f5);
                border-bottom: 1px solid var(--border-color, #ddd);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .visualization-title {
                font-size: 14px;
                font-weight: 500;
                color: var(--text-primary-color, #333);
            }

            .visualization-controls {
                display: flex;
                gap: 8px;
            }

            .control-button {
                padding: 4px 8px;
                border: 1px solid var(--border-color, #ddd);
                border-radius: 4px;
                background: var(--surface-color, #fff);
                cursor: pointer;
                font-size: 12px;
                color: var(--text-secondary-color, #666);
            }

            .control-button:hover {
                background: var(--hover-color, #f5f5f5);
            }
        `;
    }

    render() {
        return html`
            <div class="performance-monitor">
                <!-- Existierende Controls -->
                <div class="visualization-controls">
                    <button 
                        class="control-button ${this.activeVisualization === 'standard' ? 'active' : ''}"
                        @click="${() => this.activeVisualization = 'standard'}">
                        Standard
                    </button>
                    <button 
                        class="control-button ${this.activeVisualization === 'flame' ? 'active' : ''}"
                        @click="${() => this.activeVisualization = 'flame'}">
                        Flame Graph
                    </button>
                    <button 
                        class="control-button ${this.activeVisualization === 'correlation' ? 'active' : ''}"
                        @click="${() => this.activeVisualization = 'correlation'}">
                        Correlation
                    </button>
                    <button 
                        class="control-button ${this.activeVisualization === 'resources' ? 'active' : ''}"
                        @click="${() => this.activeVisualization = 'resources'}">
                        Resources
                    </button>
                </div>

                <!-- Visualization Container -->
                <div class="visualization-container">
                    <!-- Standard View (bereits existierend) -->
                    <div class="view ${this.activeVisualization === 'standard' ? '' : 'hidden'}">
                        ${this.renderMetricCards()}
                    </div>

                    <!-- Flame Graph View -->
                    <div class="view ${this.activeVisualization === 'flame' ? '' : 'hidden'}">
                        <div class="flame-graph" id="flameGraphChart"></div>
                    </div>

                    <!-- Correlation Plot View -->
                    <div class="view ${this.activeVisualization === 'correlation' ? '' : 'hidden'}">
                        <div class="correlation-plot" id="correlationPlotContainer">
                            <div class="metric-selectors">
                                <select @change="${this._updateXAxis}">
                                    ${Object.keys(this.metrics).map(key => html`
                                        <option value="${key}">${this.getMetricLabel(key)}</option>
                                    `)}
                                </select>
                                vs
                                <select @change="${this._updateYAxis}">
                                    ${Object.keys(this.metrics).map(key => html`
                                        <option value="${key}">${this.getMetricLabel(key)}</option>
                                    `)}
                                </select>
                            </div>
                        </div>
                    </div>

                    <!-- Resource Map View -->
                    <div class="view ${this.activeVisualization === 'resources' ? '' : 'hidden'}">
                        <div class="resource-map" id="resourceMapContainer"></div>
                    </div>
                </div>
            </div>
        `;
    }

    // Event Handler für Visualization Switches
    updated(changedProperties) {
        super.updated(changedProperties);
        
        if (changedProperties.has('activeVisualization')) {
            this._initActiveVisualization();
        }
    }

    _initActiveVisualization() {
        switch(this.activeVisualization) {
            case 'flame':
                this.initFlameGraph();
                break;
            case 'correlation':
                this.initCorrelationPlot();
                break;
            case 'resources':
                this.initSystemResourceMap();
                break;
        }
    }

    renderMetricCards() {
        return Object.entries(this.metrics).map(([key, value]) => {
            const isAlert = this.checkThreshold(key, value);
            return html`
                <div class="metric-card ${isAlert ? 'alert' : ''}"
                     @click="${() => this.selectedMetric = key}">
                    <div class="metric-value">
                        ${this.formatMetricValue(key, value)}
                    </div>
                    <div class="metric-label">
                        ${this.getMetricLabel(key)}
                    </div>
                </div>
            `;
        });
    }

    renderDetailedView() {
        return html`
            <div class="detailed-view">
                ${this._renderVisualizationControls()}
                
                ${this.visualizationMode === 'multi' 
                    ? this._renderMultiChartView() 
                    : this._renderSingleChartView()}
                
                <div class="compare-controls">
                    <button 
                        @click="${() => this.compareMode = !this.compareMode}"
                        class="toggle-button">
                        ${this.compareMode ? 'Disable' : 'Enable'} Compare Mode
                    </button>
                    
                    ${this.compareMode ? html`
                        <select 
                            class="metric-selector"
                            @change="${this._handleMetricAdd}">
                            <option value="">Add metric to compare...</option>
                            ${Object.keys(this.metrics).map(key => html`
                                <option value="${key}" 
                                    ?disabled="${this.comparedMetrics.includes(key) || key === this.selectedMetric}">
                                    ${this.getMetricLabel(key)}
                                </option>
                            `)}
                        </select>

                        <div class="compared-metrics">
                            ${this.comparedMetrics.map(metric => html`
                                <span class="compare-badge">
                                    ${this.getMetricLabel(metric)}
                                    <button @click="${() => this._removeMetric(metric)}">×</button>
                                </span>
                            `)}
                        </div>
                    ` : ''}
                </div>

                <div class="export-controls">
                    <select 
                        class="export-format"
                        @change="${(e) => this.exportFormat = e.target.value}">
                        <option value="json">JSON</option>
                        <option value="csv">CSV</option>
                        <option value="html">HTML Report</option>
                    </select>
                    <button @click="${this._exportData}">Export Data</button>
                </div>

                <button 
                    @click="${() => this.showStats = !this.showStats}">
                    ${this.showStats ? 'Hide' : 'Show'} Statistics
                </button>

                ${this.showStats ? this.renderStatistics() : ''}

                <div class="alerts-panel">
                    <div class="alerts-header">
                        <h4>Alert Configuration</h4>
                        <label>
                            <input 
                                type="checkbox" 
                                .checked="${this.notificationsEnabled}"
                                @change="${this._toggleNotifications}">
                            Enable Notifications
                        </label>
                    </div>

                    ${this._renderThresholdControls()}
                    ${this._renderActiveAlerts()}
                    ${this._renderAlertHistory()}
                </div>

                ${this.renderAlertTemplates()}
            </div>
        `;
    }

    _renderVisualizationControls() {
        return html`
            <div class="visualization-controls">
                ${this._renderVisualizationButtons()}
            </div>

            <div class="chart-wrapper">
                <div class="chart-container ${this.activeVisualization === 'line' ? '' : 'hidden'}"
                     id="lineChartContainer">
                    <canvas id="performanceChart"></canvas>
                    ${this._renderChartOverlay('line')}
                </div>

                <div class="chart-container ${this.activeVisualization === 'heatmap' ? '' : 'hidden'}"
                     id="heatmapContainer">
                    <canvas id="heatmapChart"></canvas>
                    ${this._renderChartOverlay('heatmap')}
                </div>

                <div class="chart-container ${this.activeVisualization === 'waterfall' ? '' : 'hidden'}"
                     id="waterfallContainer">
                    <canvas id="waterfallChart"></canvas>
                    ${this._renderChartOverlay('waterfall')}
                </div>

                <div class="chart-container ${this.activeVisualization === 'stacktrace' ? '' : 'hidden'}"
                     id="stackTraceContainer">
                    ${this._renderChartOverlay('stacktrace')}
                </div>

                <div class="chart-container ${this.activeVisualization === 'flamegraph' ? '' : 'hidden'}"
                     id="flameGraphChart">
                    ${this._renderChartOverlay('flamegraph')}
                </div>
            </div>

            ${this._renderVisualizationControls()}
        `;
    }

    _renderVisualizationButtons() {
        const visualizations = [
            { id: 'line', label: 'Timeline', icon: this._getLineChartIcon() },
            { id: 'heatmap', label: 'Heatmap', icon: this._getHeatmapIcon() },
            { id: 'waterfall', label: 'Waterfall', icon: this._getWaterfallIcon() },
            { id: 'stacktrace', label: 'Stack Trace', icon: this._getStackTraceIcon() },
            { id: 'flamegraph', label: 'Flame Graph', icon: this._getFlameGraphIcon() }
        ];

        return html`
            ${visualizations.map(viz => html`
                <button class="viz-button ${this.activeVisualization === viz.id ? 'active' : ''}"
                        @click="${() => this._switchVisualization(viz.id)}">
                    ${viz.icon}
                    ${viz.label}
                </button>
            `)}
        `;
    }

    _renderChartOverlay(chartType) {
        return html`
            <div class="chart-overlay">
                <div class="loading-spinner"></div>
            </div>
        `;
    }

    async _switchVisualization(type) {
        if (type === this.activeVisualization) return;

        const container = this.shadowRoot.querySelector(`#${type}Container`);
        container.classList.add('loading');

        this.activeVisualization = type;
        await this._initializeVisualization(type);

        setTimeout(() => {
            container.classList.remove('loading');
        }, 500);
    }

    async _initializeVisualization(type) {
        switch (type) {
            case 'heatmap':
                await this.initHeatmap();
                break;
            case 'waterfall':
                await this.initWaterfall();
                break;
            case 'stacktrace':
                await this.initStackTrace();
                break;
            case 'flamegraph':
                await this.initFlameGraph();
                break;
            default:
                await this.initCharts();
        }
    }

    // Icon-Definitionen als SVG
    _getLineChartIcon() {
        return html`<svg viewBox="0 0 24 24">
            <path d="M16,11.78L20.24,4.45L21.97,5.45L16.74,14.5L10.23,10.75L5.46,19H22V21H2V3H4V17.54L9.5,8L16,11.78Z" />
        </svg>`;
    }

    // ... weitere Icon-Definitionen ...

    _renderMultiChartView() {
        return html`
            <div class="multi-chart-container">
                ${Object.keys(this.metrics).map(metric => html`
                    <div class="chart-container">
                        <h4>${this.getMetricLabel(metric)}</h4>
                        <canvas id="chart-${metric}"></canvas>
                    </div>
                `)}
            </div>
        `;
    }

    _renderSingleChartView() {
        return html`
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        `;
    }

    _updateChartType(type) {
        this.chartType = type;
        this._updateAllCharts();
    }

    _updateAllCharts() {
        if (this.visualizationMode === 'multi') {
            Object.keys(this.metrics).forEach(metric => {
                this._updateSingleChart(metric);
            });
        } else {
            this.updateChart();
        }
    }

    _updateSingleChart(metric) {
        const canvas = this.shadowRoot.querySelector(`#chart-${metric}`);
        if (!canvas) return;

        if (this.charts[metric]) {
            this.charts[metric].destroy();
        }

        const ctx = canvas.getContext('2d');
        const data = this._prepareChartData(metric);
        
        this.charts[metric] = new Chart(ctx, {
            type: this._getChartConfig(this.chartType).type,
            data: this._createDataset(data, metric),
            options: this._getChartOptions(metric)
        });
    }

    _getChartConfig(type) {
        const configs = {
            line: {
                type: 'line',
                fill: false
            },
            bar: {
                type: 'bar'
            },
            scatter: {
                type: 'scatter'
            },
            area: {
                type: 'line',
                fill: true
            }
        };
        return configs[type] || configs.line;
    }

    _createDataset(data, metric) {
        const config = this._getChartConfig(this.chartType);
        return {
            labels: data.labels,
            datasets: [{
                label: this.getMetricLabel(metric),
                data: data.values,
                borderColor: this._getMetricColor(metric),
                backgroundColor: config.fill 
                    ? this._getMetricColor(metric, 0.2) 
                    : this._getMetricColor(metric),
                fill: config.fill,
                tension: 0.4
            }]
        };
    }

    _getMetricColor(metric, alpha = 1) {
        const colors = {
            avgResponseTime: `rgba(33, 150, 243, ${alpha})`,
            errorRate: `rgba(244, 67, 54, ${alpha})`,
            cacheHitRate: `rgba(76, 175, 80, ${alpha})`,
            memoryUsage: `rgba(255, 193, 7, ${alpha})`
        };
        return colors[metric] || `rgba(158, 158, 158, ${alpha})`;
    }

    _getChartOptions(metric) {
        const theme = this._getChartTheme();
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: theme.gridColor
                    },
                    ticks: {
                        color: theme.textColor
                    }
                },
                x: {
                    grid: {
                        color: theme.gridColor
                    },
                    ticks: {
                        color: theme.textColor
                    }
                }
            },
            plugins: {
                legend: {
                    display: this.compareMode,
                    labels: {
                        color: theme.textColor
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            return `${context.dataset.label}: ${this.formatMetricValue(
                                context.dataset.metricKey,
                                context.parsed.y
                            )}`;
                        }
                    }
                }
            }
        };
    }

    _prepareChartData(metric) {
        const timeRange = this.getTimeRangeInMs();
        const now = Date.now();
        const filteredData = this.historicalData.filter(entry => 
            new Date(entry.timestamp).getTime() > now - timeRange
        );

        const labels = filteredData.map(entry => 
            new Date(entry.timestamp).toLocaleTimeString()
        );

        const values = {
            [metric]: filteredData.map(entry => entry[metric])
        };

        return { labels, values };
    }

    _handleMetricAdd(e) {
        const metric = e.target.value;
        if (metric && !this.comparedMetrics.includes(metric)) {
            this.comparedMetrics = [...this.comparedMetrics, metric];
            e.target.value = ''; // Reset select
        }
    }

    _removeMetric(metric) {
        this.comparedMetrics = this.comparedMetrics.filter(m => m !== metric);
    }

    getMetricUnit(metric) {
        const units = {
            avgResponseTime: 'Milliseconds',
            messageCount: 'Count',
            tokenUsage: 'Tokens',
            cacheHitRate: 'Percentage',
            memoryUsage: 'MB',
            errorRate: 'Percentage',
            apiCallsPerMinute: 'Calls/min',
            wsLatency: 'Milliseconds'
        };
        return units[metric] || '';
    }

    renderStatistics() {
        const stats = this._calculateStats();
        return html`
            <div class="stats-panel">
                <h4>Statistical Analysis</h4>
                <div class="stats-grid">
                    ${Object.entries(stats).map(([key, value]) => html`
                        <div class="stat-item">
                            <div class="stat-label">${key}</div>
                            <div class="stat-value">${this.formatMetricValue(this.selectedMetric, value)}</div>
                        </div>
                    `)}
                </div>
            </div>
        `;
    }

    _calculateStats() {
        const values = this.historicalData
            .map(entry => entry[this.selectedMetric])
            .filter(value => value !== undefined);

        if (values.length === 0) return {};

        const sorted = [...values].sort((a, b) => a - b);
        const sum = values.reduce((a, b) => a + b, 0);
        const mean = sum / values.length;
        const median = sorted[Math.floor(values.length / 2)];
        const min = sorted[0];
        const max = sorted[values.length - 1];

        // Standardabweichung
        const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);

        // 95. Perzentil
        const p95Index = Math.floor(values.length * 0.95);
        const p95 = sorted[p95Index];

        return {
            'Average': mean,
            'Median': median,
            'Min': min,
            'Max': max,
            'Std Dev': stdDev,
            'P95': p95
        };
    }

    _exportData() {
        const data = {
            timestamp: new Date().toISOString(),
            metric: this.selectedMetric,
            timeRange: this.timeRange,
            statistics: this._calculateStats(),
            historicalData: this.historicalData.map(entry => ({
                timestamp: entry.timestamp,
                value: entry[this.selectedMetric]
            }))
        };

        switch (this.exportFormat) {
            case 'json':
                this._downloadFile(
                    JSON.stringify(data, null, 2),
                    'performance-data.json',
                    'application/json'
                );
                break;
            case 'csv':
                this._exportCSV(data);
                break;
            case 'html':
                this._exportHTML(data);
                break;
        }
    }

    _exportCSV(data) {
        const headers = ['Timestamp', 'Value'];
        const rows = data.historicalData.map(entry => [
            entry.timestamp,
            entry.value
        ]);

        const csv = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        this._downloadFile(csv, 'performance-data.csv', 'text/csv');
    }

    _exportHTML(data) {
        const html = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Performance Report - ${this.getMetricLabel(this.selectedMetric)}</title>
                <style>
                    body { font-family: sans-serif; padding: 20px; }
                    .stats { margin: 20px 0; }
                    table { width: 100%; border-collapse: collapse; }
                    th, td { padding: 8px; border: 1px solid #ddd; }
                    th { background: #f5f5f5; }
                </style>
            </head>
            <body>
                <h1>Performance Report</h1>
                <h2>${this.getMetricLabel(this.selectedMetric)}</h2>
                <div class="stats">
                    <h3>Statistics</h3>
                    <table>
                        ${Object.entries(data.statistics)
                            .map(([key, value]) => `
                                <tr>
                                    <th>${key}</th>
                                    <td>${this.formatMetricValue(this.selectedMetric, value)}</td>
                                </tr>
                            `).join('')}
                    </table>
                </div>
                <div class="history">
                    <h3>Historical Data</h3>
                    <table>
                        <tr>
                            <th>Timestamp</th>
                            <th>Value</th>
                        </tr>
                        ${data.historicalData
                            .map(entry => `
                                <tr>
                                    <td>${entry.timestamp}</td>
                                    <td>${this.formatMetricValue(this.selectedMetric, entry.value)}</td>
                                </tr>
                            `).join('')}
                    </table>
                </div>
            </body>
            </html>
        `;

        this._downloadFile(html, 'performance-report.html', 'text/html');
    }

    _downloadFile(content, filename, type) {
        const blob = new Blob([content], { type });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    _renderThresholdControls() {
        return html`
            <div class="threshold-controls">
                <h5>Thresholds for ${this.getMetricLabel(this.selectedMetric)}</h5>
                ${this.thresholds[this.selectedMetric] ? html`
                    <div class="threshold-control">
                        <div class="threshold-input">
                            <label>Warning:</label>
                            <input 
                                type="number"
                                .value="${this.thresholds[this.selectedMetric].warning}"
                                @change="${(e) => this._updateThreshold('warning', e.target.value)}">
                        </div>
                        <div class="threshold-input">
                            <label>Critical:</label>
                            <input 
                                type="number"
                                .value="${this.thresholds[this.selectedMetric].critical}"
                                @change="${(e) => this._updateThreshold('critical', e.target.value)}">
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    _renderActiveAlerts() {
        return html`
            <div class="active-alerts">
                <h5>Active Alerts</h5>
                ${this.alerts.map(alert => html`
                    <div class="alert-item alert-${alert.level}">
                        <span>${alert.message}</span>
                        <span>${this._formatTimestamp(alert.timestamp)}</span>
                    </div>
                `)}
            </div>
        `;
    }

    _renderAlertHistory() {
        return html`
            <div class="alert-history">
                <h5>Alert History</h5>
                ${this.alertHistory.slice(-5).map(alert => html`
                    <div class="alert-item alert-${alert.level}">
                        <span>${alert.message}</span>
                        <span>${this._formatTimestamp(alert.timestamp)}</span>
                    </div>
                `)}
            </div>
        `;
    }

    async _requestNotificationPermission() {
        if ('Notification' in window) {
            const permission = await Notification.requestPermission();
            this.notificationsEnabled = permission === 'granted';
        }
    }

    _toggleNotifications(e) {
        this.notificationsEnabled = e.target.checked;
        if (this.notificationsEnabled) {
            this._requestNotificationPermission();
        }
    }

    _startAlertMonitoring() {
        setInterval(() => {
            Object.entries(this.metrics).forEach(([metric, value]) => {
                this._checkThresholds(metric, value);
            });
        }, 5000); // Prüfe alle 5 Sekunden
    }

    _checkThresholds(metric, value) {
        const thresholds = this.thresholds[metric];
        if (!thresholds) return;

        let newAlert = null;

        if (value >= thresholds.critical) {
            newAlert = {
                level: 'critical',
                message: `${this.getMetricLabel(metric)} is critical: ${this.formatMetricValue(metric, value)}`,
                timestamp: new Date(),
                metric
            };
        } else if (value >= thresholds.warning) {
            newAlert = {
                level: 'warning',
                message: `${this.getMetricLabel(metric)} is high: ${this.formatMetricValue(metric, value)}`,
                timestamp: new Date(),
                metric
            };
        }

        if (newAlert) {
            this._addAlert(newAlert);
        } else {
            // Entferne gelöste Alerts
            this.alerts = this.alerts.filter(alert => alert.metric !== metric);
        }
    }

    _addAlert(alert) {
        // Prüfe, ob ein ähnlicher Alert bereits existiert
        const existingAlert = this.alerts.find(a => a.metric === alert.metric);
        if (!existingAlert) {
            this.alerts = [...this.alerts, alert];
            this.alertHistory = [...this.alertHistory, alert];
            
            // Beschränke History auf die letzten 100 Einträge
            if (this.alertHistory.length > 100) {
                this.alertHistory = this.alertHistory.slice(-100);
            }

            // Sende Desktop-Benachrichtigung
            if (this.notificationsEnabled) {
                this._sendNotification(alert);
            }
        }
    }

    _sendNotification(alert) {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('Performance Alert', {
                body: alert.message,
                icon: '/static/img/alert-icon.png'
            });
        }
    }

    _updateThreshold(level, value) {
        this.thresholds = {
            ...this.thresholds,
            [this.selectedMetric]: {
                ...this.thresholds[this.selectedMetric],
                [level]: Number(value)
            }
        };
    }

    _formatTimestamp(timestamp) {
        return new Date(timestamp).toLocaleTimeString();
    }

    startMetricsCollection() {
        // Polling alle 5 Sekunden
        setInterval(async () => {
            try {
                const response = await fetch('/api/debug/performance');
                const data = await response.json();
                this.updateMetrics(data);
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
            }
        }, 5000);
    }

    updateMetrics(data) {
        this.metrics = { ...this.metrics, ...data };
        this.historicalData.push({
            timestamp: new Date(),
            ...data
        });

        // Behalte nur Daten für den ausgewählten Zeitraum
        this.pruneHistoricalData();
    }

    pruneHistoricalData() {
        const now = new Date();
        const threshold = new Date(now - this.getTimeRangeInMs());
        this.historicalData = this.historicalData.filter(
            entry => new Date(entry.timestamp) > threshold
        );
    }

    getTimeRangeInMs() {
        const ranges = {
            '1h': 60 * 60 * 1000,
            '24h': 24 * 60 * 60 * 1000,
            '7d': 7 * 24 * 60 * 60 * 1000
        };
        return ranges[this.timeRange];
    }

    checkThreshold(metric, value) {
        if (!(metric in this.alertThresholds)) return false;
        return value > this.alertThresholds[metric];
    }

    formatMetricValue(metric, value) {
        const formatters = {
            avgResponseTime: v => `${v.toFixed(0)}ms`,
            messageCount: v => v.toLocaleString(),
            tokenUsage: v => v.toLocaleString(),
            cacheHitRate: v => `${v.toFixed(1)}%`,
            memoryUsage: v => `${(v / 1024 / 1024).toFixed(1)}MB`,
            errorRate: v => `${v.toFixed(1)}%`,
            apiCallsPerMinute: v => v.toFixed(1),
            wsLatency: v => `${v.toFixed(0)}ms`
        };
        return formatters[metric] ? formatters[metric](value) : value;
    }

    getMetricLabel(key) {
        const labels = {
            avgResponseTime: 'Avg Response Time',
            messageCount: 'Message Count',
            tokenUsage: 'Token Usage',
            cacheHitRate: 'Cache Hit Rate',
            dbQueryCount: 'DB Queries',
            memoryUsage: 'Memory Usage',
            activeConnections: 'Active Connections',
            wsLatency: 'WebSocket Latency',
            apiCallsPerMinute: 'API Calls/min',
            errorRate: 'Error Rate'
        };
        return labels[key] || key;
    }

    updated(changedProperties) {
        super.updated(changedProperties);
        
        if (this.isExpanded && 
            (changedProperties.has('selectedMetric') || 
             changedProperties.has('historicalData'))) {
            this.updateChart();
        }
    }

    updateChart() {
        const canvas = this.shadowRoot.querySelector('#performanceChart');
        if (!canvas || !this.charts.performance) return;

        const config = this._getChartConfig();
        
        // Update existing chart
        this.charts.performance.options = {
            ...this.charts.performance.options,
            ...config.options
        };
        
        this.charts.performance.update('none');
    }

    _getChartConfig() {
        return {
            options: {
                responsive: true,
                interaction: {
                    mode: 'nearest',
                    intersect: false,
                    axis: 'x'
                },
                plugins: {
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true,
                                mode: this.zoomMode
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: this.zoomMode,
                            drag: {
                                enabled: true,
                                borderColor: 'rgb(54, 162, 235)',
                                borderWidth: 1,
                                backgroundColor: 'rgba(54, 162, 235, 0.3)'
                            }
                        },
                        pan: {
                            enabled: true,
                            mode: this.zoomMode
                        }
                    },
                    tooltip: {
                        enabled: true,
                        position: 'nearest',
                        external: this._externalTooltipHandler.bind(this)
                    },
                    annotation: {
                        annotations: this._getAnnotationConfig()
                    }
                }
            }
        };
    }

    _externalTooltipHandler(context) {
        const { chart, tooltip } = context;
        const position = chart.canvas.getBoundingClientRect();

        // Remove existing tooltip
        const existingTooltip = this.shadowRoot.querySelector('.tooltip-detail');
        if (existingTooltip) {
            existingTooltip.remove();
        }

        // If no tooltip, return
        if (tooltip.opacity === 0) {
            this.selectedDataPoint = null;
            return;
        }

        // Create new tooltip
        const tooltipEl = document.createElement('div');
        tooltipEl.classList.add('tooltip-detail');

        // Position the tooltip
        const left = position.left + tooltip.caretX;
        const top = position.top + tooltip.caretY;
        tooltipEl.style.left = `${left}px`;
        tooltipEl.style.top = `${top}px`;

        // Get data point
        const dataPoint = this._getDataPointFromTooltip(tooltip);
        this.selectedDataPoint = dataPoint;

        // Render tooltip content
        tooltipEl.innerHTML = this._getTooltipContent(dataPoint);

        // Add to shadow DOM
        this.shadowRoot.appendChild(tooltipEl);
    }

    _getDataPointFromTooltip(tooltip) {
        const datasetIndex = tooltip.dataPoints[0].datasetIndex;
        const index = tooltip.dataPoints[0].index;
        const dataset = this.charts.performance.data.datasets[datasetIndex];
        
        return {
            label: dataset.label,
            value: dataset.data[index],
            timestamp: this.charts.performance.data.labels[index],
            metric: dataset.metricKey
        };
    }

    _getTooltipContent(dataPoint) {
        if (!dataPoint) return '';

        const timestamp = format(new Date(dataPoint.timestamp), 'PPpp');
        const value = this.formatMetricValue(dataPoint.metric, dataPoint.value);
        const stats = this._getPointStatistics(dataPoint);

        return `
            <div class="tooltip-content">
                <h4>${dataPoint.label}</h4>
                <p><strong>Time:</strong> ${timestamp}</p>
                <p><strong>Value:</strong> ${value}</p>
                <div class="tooltip-stats">
                    <p><strong>Average:</strong> ${stats.average}</p>
                    <p><strong>Trend:</strong> ${stats.trend}</p>
                </div>
                ${this._getAlertInfo(dataPoint)}
            </div>
        `;
    }

    _getPointStatistics(dataPoint) {
        // Calculate statistics for the surrounding data points
        const windowSize = 5; // Look at 5 points before and after
        const dataset = this.charts.performance.data.datasets[0];
        const index = dataset.data.indexOf(dataPoint.value);
        
        const window = dataset.data.slice(
            Math.max(0, index - windowSize),
            Math.min(dataset.data.length, index + windowSize + 1)
        );

        const average = this.formatMetricValue(
            dataPoint.metric,
            window.reduce((a, b) => a + b, 0) / window.length
        );

        const trend = this._calculateTrend(window);

        return { average, trend };
    }

    _calculateTrend(values) {
        const first = values[0];
        const last = values[values.length - 1];
        const diff = last - first;

        if (Math.abs(diff) < 0.001) return 'Stable';
        return diff > 0 ? 'Increasing' : 'Decreasing';
    }

    _getAlertInfo(dataPoint) {
        const alerts = this.alerts.filter(alert => 
            alert.metric === dataPoint.metric &&
            Math.abs(new Date(alert.timestamp) - new Date(dataPoint.timestamp)) < 60000
        );

        if (!alerts.length) return '';

        return `
            <div class="tooltip-alerts">
                <strong>Alerts:</strong>
                ${alerts.map(alert => html`
                    <div class="alert-${alert.level}">
                        ${alert.message}
                    </div>
                `).join('')}
            </div>
        `;
    }

    _getAnnotationConfig() {
        return this.annotations.map(annotation => ({
            type: 'line',
            borderColor: annotation.color || 'rgba(255, 99, 132, 0.5)',
            borderWidth: 2,
            label: {
                content: annotation.label,
                enabled: true
            },
            scaleID: 'x',
            value: annotation.timestamp
        }));
    }

    _addAnnotation(event) {
        const annotation = {
            timestamp: new Date(event.timestamp).getTime(),
            label: event.label || 'Custom Annotation',
            color: event.color || 'rgba(255, 99, 132, 0.5)'
        };

        this.annotations = [...this.annotations, annotation];
        this.updateChart();
    }

    _initWorker() {
        try {
            this.worker = new Worker('/static/js/bruno/performance-worker.js');
            this.worker.onmessage = (e) => {
                const { type, data } = e.data;
                switch (type) {
                    case 'aggregatedData':
                        this._handleAggregatedData(data);
                        break;
                    case 'processedMetrics':
                        this._updateMetricsDisplay(data);
                        break;
                }
            };
            
            this.worker.onerror = (error) => {
                console.error('Performance Worker Error:', error);
                // Fallback zur synchronen Verarbeitung
                this._processSynchronously = true;
            };
        } catch (error) {
            console.warn('Worker initialization failed, falling back to synchronous processing:', error);
            this._processSynchronously = true;
        }
    }

    _processData(data) {
        if (this._processSynchronously) {
            // Synchrone Verarbeitung als Fallback
            return this._processSynchronously(data);
        }

        // Asynchrone Verarbeitung via Worker
        this.worker.postMessage({
            type: 'processData',
            data: {
                metrics: data,
                timeRange: this.timeRange,
                resolution: this.dataResolution
            }
        });
    }

    _handleAggregatedData(data) {
        this.historicalData = data;
        this.requestUpdate();
    }

    // Fallback für synchrone Verarbeitung
    _processSynchronously(data) {
        // Einfache synchrone Datenverarbeitung
        return data;
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        if (this.worker) {
            this.worker.terminate();
        }
        if (this.exportManager) {
            this.exportManager.destroy();
        }
    }

    renderAlertTemplates() {
        return html`
            <div class="template-selector">
                <h4>Alert Templates</h4>
                
                <select 
                    .value="${this.selectedTemplate}"
                    @change="${this._handleTemplateSelection}">
                    <option value="">Select a template...</option>
                    ${Object.entries(this.alertTemplateManager.getAllTemplates()).map(
                        ([id, template]) => html`
                            <option value="${id}">${template.name}</option>
                        `
                    )}
                </select>

                <div class="active-templates">
                    ${this.activeTemplates.map(templateId => {
                        const template = this.alertTemplateManager.getTemplate(templateId);
                        return html`
                            <div class="template-card">
                                <h5>${template.name}</h5>
                                <p>${template.description}</p>
                                
                                <div class="threshold-controls">
                                    <label>Warning Threshold:
                                        <input 
                                            type="number"
                                            .value="${template.conditions.warning.threshold}"
                                            @change="${(e) => this._updateTemplateThreshold(
                                                templateId, 
                                                'warning', 
                                                e.target.value
                                            )}">
                                    </label>
                                    <label>Critical Threshold:
                                        <input 
                                            type="number"
                                            .value="${template.conditions.critical.threshold}"
                                            @change="${(e) => this._updateTemplateThreshold(
                                                templateId, 
                                                'critical', 
                                                e.target.value
                                            )}">
                                    </label>
                                </div>

                                <div class="template-suggestions">
                                    <strong>Suggestions:</strong>
                                    <ul>
                                        ${template.suggestions.map(
                                            suggestion => html`<li>${suggestion}</li>`
                                        )}
                                    </ul>
                                </div>

                                <button 
                                    @click="${() => this._removeTemplate(templateId)}">
                                    Remove Template
                                </button>
                            </div>
                        `;
                    })}
                </div>
            </div>
        `;
    }

    _handleTemplateSelection(e) {
        const templateId = e.target.value;
        if (templateId && !this.activeTemplates.includes(templateId)) {
            this.activeTemplates = [...this.activeTemplates, templateId];
            this._applyTemplate(templateId);
            e.target.value = ''; // Reset selection
        }
    }

    _applyTemplate(templateId) {
        const template = this.alertTemplateManager.getTemplate(templateId);
        if (template) {
            // Aktualisiere Schwellenwerte für die entsprechende Metrik
            this.thresholds = {
                ...this.thresholds,
                [template.metric]: {
                    warning: template.conditions.warning.threshold,
                    critical: template.conditions.critical.threshold
                }
            };
            
            // Starte Überwachung
            this._startMonitoring(template);
        }
    }

    _startMonitoring(template) {
        const checkInterval = setInterval(() => {
            const currentValue = this.metrics[template.metric];
            if (currentValue >= template.conditions.critical.threshold) {
                this._createAlert({
                    level: 'critical',
                    message: `${template.name}: ${this.getMetricLabel(template.metric)} is critical`,
                    metric: template.metric,
                    value: currentValue,
                    suggestions: template.suggestions
                });
            } else if (currentValue >= template.conditions.warning.threshold) {
                this._createAlert({
                    level: 'warning',
                    message: `${template.name}: ${this.getMetricLabel(template.metric)} is high`,
                    metric: template.metric,
                    value: currentValue,
                    suggestions: template.suggestions
                });
            }
        }, 5000); // Prüfe alle 5 Sekunden

        // Speichere Interval-ID für späteres Cleanup
        this._monitoringIntervals = this._monitoringIntervals || new Map();
        this._monitoringIntervals.set(template.id, checkInterval);
    }

    _removeTemplate(templateId) {
        this.activeTemplates = this.activeTemplates.filter(id => id !== templateId);
        
        // Cleanup Monitoring
        if (this._monitoringIntervals?.has(templateId)) {
            clearInterval(this._monitoringIntervals.get(templateId));
            this._monitoringIntervals.delete(templateId);
        }
    }

    _updateTemplateThreshold(templateId, level, value) {
        const template = this.alertTemplateManager.getTemplate(templateId);
        if (template) {
            template.conditions[level].threshold = Number(value);
            this._applyTemplate(templateId); // Aktualisiere Überwachung
        }
    }

    renderExportPanel() {
        return html`
            <div class="export-panel">
                <h4>Export Data</h4>
                
                <div class="export-controls">
                    <select 
                        .value="${this.exportFormat}"
                        @change="${(e) => this.exportFormat = e.target.value}">
                        <option value="json">JSON</option>
                        <option value="csv">CSV</option>
                        <option value="excel">Excel</option>
                        <option value="pdf">PDF Report</option>
                    </select>

                    <button 
                        @click="${this._handleExport}"
                        ?disabled="${this.exportInProgress}">
                        ${this.exportInProgress ? html`
                            <span class="spinner"></span>
                        ` : 'Export'}
                    </button>
                </div>

                <div class="export-options">
                    ${this._renderExportOptions()}
                </div>

                ${this.exportInProgress ? html`
                    <div class="export-progress">
                        Processing data...
                    </div>
                ` : ''}

                ${this.exportError ? html`
                    <div class="export-error">
                        ${this.exportError}
                    </div>
                ` : ''}
            </div>
        `;
    }

    _renderExportOptions() {
        return html`
            <div class="option-group">
                <h5>Time Range</h5>
                <select @change="${(e) => this._updateExportOption('timeRange', e.target.value)}">
                    <option value="1h">Last Hour</option>
                    <option value="24h">Last 24 Hours</option>
                    <option value="7d">Last 7 Days</option>
                    <option value="30d">Last 30 Days</option>
                    <option value="custom">Custom Range</option>
                </select>
            </div>

            <div class="option-group">
                <h5>Resolution</h5>
                <select @change="${(e) => this._updateExportOption('resolution', e.target.value)}">
                    <option value="raw">Raw Data</option>
                    <option value="1m">1 Minute</option>
                    <option value="5m">5 Minutes</option>
                    <option value="1h">1 Hour</option>
                </select>
            </div>

            <div class="option-group">
                <h5>Metrics</h5>
                ${Object.keys(this.metrics).map(metric => html`
                    <label>
                        <input 
                            type="checkbox" 
                            .checked="${this._isMetricSelected(metric)}"
                            @change="${(e) => this._toggleMetric(metric, e.target.checked)}">
                        ${this.getMetricLabel(metric)}
                    </label>
                `)}
            </div>
        `;
    }

    async _handleExport() {
        try {
            this.exportError = '';
            this.exportInProgress = true;

            const options = {
                timeRange: this.exportTimeRange,
                resolution: this.exportResolution,
                metrics: this.selectedMetrics,
                format: this.exportFormat
            };

            const data = this._prepareExportData();
            await this.exportManager.exportData(data, this.exportFormat, options);

        } catch (error) {
            console.error('Export error:', error);
            this.exportError = `Export failed: ${error.message}`;
        } finally {
            this.exportInProgress = false;
        }
    }

    _prepareExportData() {
        return {
            metadata: {
                exportDate: new Date().toISOString(),
                timeRange: this.exportTimeRange,
                resolution: this.exportResolution,
                metrics: this.selectedMetrics
            },
            metrics: this.historicalData,
            alerts: this.alertHistory,
            statistics: this._calculateStats()
        };
    }

    _updateExportOption(option, value) {
        this[`export${option.charAt(0).toUpperCase() + option.slice(1)}`] = value;
    }

    _isMetricSelected(metric) {
        return this.selectedMetrics.includes(metric);
    }

    _toggleMetric(metric, checked) {
        if (checked) {
            this.selectedMetrics = [...this.selectedMetrics, metric];
        } else {
            this.selectedMetrics = this.selectedMetrics.filter(m => m !== metric);
        }
    }

    renderChartControls() {
        return html`
            <div class="zoom-controls">
                <button 
                    class="zoom-button ${this.zoomMode === 'x' ? 'active' : ''}"
                    @click="${() => this.zoomMode = 'x'}">
                    <span>⟷</span> X Zoom
                </button>
                <button 
                    class="zoom-button ${this.zoomMode === 'y' ? 'active' : ''}"
                    @click="${() => this.zoomMode = 'y'}">
                    <span>↕</span> Y Zoom
                </button>
                <button 
                    class="zoom-button ${this.zoomMode === 'xy' ? 'active' : ''}"
                    @click="${() => this.zoomMode = 'xy'}">
                    <span>⤡</span> XY Zoom
                </button>
                <button 
                    class="zoom-button"
                    @click="${this._resetZoom}">
                    <span>↺</span> Reset
                </button>
            </div>
        `;
    }

    _handleChartClick(event) {
        const points = this.charts.performance.getElementsAtEventForMode(
            event, 
            'nearest', 
            { intersect: true },
            false
        );

        if (points.length) {
            const point = points[0];
            const data = this._getDataPointFromChart(point);
            
            if (this.comparisonMode) {
                this._handleComparisonPoint(data);
            } else {
                this._showPointDetail(data);
            }
        }
    }

    _handleComparisonPoint(data) {
        if (!this.comparisonPoints) {
            this.comparisonPoints = [data];
        } else if (this.comparisonPoints.length === 1) {
            this.comparisonPoints.push(data);
            this._showComparison();
        } else {
            this.comparisonPoints = [data];
        }
        this.requestUpdate();
    }

    _showComparison() {
        if (this.comparisonPoints.length !== 2) return;

        const [point1, point2] = this.comparisonPoints;
        const diff = point2.value - point1.value;
        const percentChange = (diff / point1.value) * 100;

        const comparisonDetail = document.createElement('div');
        comparisonDetail.classList.add('tooltip-detail', 'comparison-detail');
        comparisonDetail.innerHTML = `
            <div class="tooltip-content">
                <h4>Comparison</h4>
                <p><strong>Time Range:</strong> ${this._formatTimeRange(point1.timestamp, point2.timestamp)}</p>
                <p><strong>Change:</strong> ${this.formatMetricValue(point1.metric, diff)}</p>
                <p><strong>Percent Change:</strong> ${percentChange.toFixed(2)}%</p>
            </div>
        `;

        this.shadowRoot.appendChild(comparisonDetail);
    }

    _formatTimeRange(t1, t2) {
        const d1 = new Date(t1);
        const d2 = new Date(t2);
        return `${format(d1, 'PPpp')} - ${format(d2, 'PPpp')}`;
    }

    _resetZoom() {
        if (this.charts.performance) {
            this.charts.performance.resetZoom();
        }
    }

    _initMemoryManager() {
        this._memoryLimit = 50 * 1024 * 1024; // 50MB
        this._gcThreshold = 0.8; // 80% des Limits
        this._dataRetentionTime = 7 * 24 * 60 * 60 * 1000; // 7 Tage
        
        // Periodische Überprüfung des Speicherverbrauchs
        setInterval(() => this._checkMemoryUsage(), 60000);
    }

    _checkMemoryUsage() {
        const currentSize = this._estimateDataSize();
        if (currentSize > this._memoryLimit * this._gcThreshold) {
            this._performGarbageCollection();
        }
    }

    _estimateDataSize() {
        return new Blob([JSON.stringify(this.historicalData)]).size;
    }

    _performGarbageCollection() {
        const cutoffTime = Date.now() - this._dataRetentionTime;
        this.historicalData = this.historicalData.filter(
            item => new Date(item.timestamp).getTime() > cutoffTime
        );
        this.requestUpdate();
    }

    renderMetricsTable() {
        return html`
            <lit-virtualizer
                .items=${this._getVisibleData()}
                .renderItem=${(item) => this._renderMetricRow(item)}
                scroller
                .scrollTarget=${this}
                class="metrics-table"
            ></lit-virtualizer>
        `;
    }

    _getVisibleData() {
        return this.historicalData.filter(item => {
            const timestamp = new Date(item.timestamp).getTime();
            return timestamp >= this.visibleTimeRange.start && 
                   timestamp <= this.visibleTimeRange.end;
        });
    }

    _renderMetricRow(item) {
        return html`
            <div class="metric-row">
                <div class="metric-timestamp">
                    ${this._formatTimestamp(item.timestamp)}
                </div>
                <div class="metric-value ${this._getValueClass(item.value)}">
                    ${this._formatValue(item.value)}
                </div>
                ${this._renderMetricDetails(item)}
            </div>
        `;
    }

    /**
     * Lädt einen einzelnen Daten-Chunk
     */
    async _loadChunk(chunkId) {
        if (this.activeChunks.includes(chunkId)) {
            return; // Chunk bereits geladen
        }

        try {
            // Chunk-Zeitbereich berechnen
            const chunkStart = chunkId * this.dataChunkSize * 1000;
            const chunkEnd = chunkStart + (this.dataChunkSize * 1000);

            // Prüfe Cache
            const cachedChunk = await this._getChunkFromCache(chunkId);
            if (cachedChunk) {
                this._processChunkData(cachedChunk, chunkId);
                return;
            }

            // Lade Daten vom Server
            const chunkData = await this._fetchChunkFromServer(chunkStart, chunkEnd);
            
            // Verarbeite und cache Daten
            this._processChunkData(chunkData, chunkId);
            this._cacheChunk(chunkId, chunkData);

        } catch (error) {
            console.error(`Error loading chunk ${chunkId}:`, error);
            // Markiere Chunk als fehlgeschlagen für Retry-Logik
            this._failedChunks.add(chunkId);
        }
    }

    /**
     * Chunk Caching System
     */
    async _getChunkFromCache(chunkId) {
        try {
            // In-Memory Cache
            if (this._memoryCache.has(chunkId)) {
                return this._memoryCache.get(chunkId);
            }

            // IndexedDB Cache
            if ('indexedDB' in window) {
                const db = await this._getDatabase();
                const transaction = db.transaction(['chunks'], 'readonly');
                const store = transaction.objectStore('chunks');
                return await store.get(chunkId);
            }
        } catch (error) {
            console.warn('Cache read error:', error);
            return null;
        }
    }

    async _cacheChunk(chunkId, data) {
        try {
            // In-Memory Cache mit LRU
            this._memoryCache.set(chunkId, data);
            
            // Prüfe Memory Cache Größe
            if (this._memoryCache.size > this._maxMemoryCacheSize) {
                const oldestKey = this._memoryCache.keys().next().value;
                this._memoryCache.delete(oldestKey);
            }

            // IndexedDB Cache
            if ('indexedDB' in window) {
                const db = await this._getDatabase();
                const transaction = db.transaction(['chunks'], 'readwrite');
                const store = transaction.objectStore('chunks');
                await store.put(data, chunkId);
            }
        } catch (error) {
            console.warn('Cache write error:', error);
        }
    }

    /**
     * Chunk Daten Verarbeitung
     */
    _processChunkData(data, chunkId) {
        // Daten normalisieren und aggregieren
        const processedData = this._normalizeChunkData(data);
        
        // Update aktive Chunks
        this.activeChunks = [...this.activeChunks, chunkId].sort();
        
        // Merge in historicalData
        this._mergeChunkData(processedData);
        
        // Trigger Update
        this.requestUpdate();
    }

    _normalizeChunkData(data) {
        return data.map(item => ({
            ...item,
            timestamp: new Date(item.timestamp).getTime(),
            value: Number(item.value)
        })).sort((a, b) => a.timestamp - b.timestamp);
    }

    _mergeChunkData(newData) {
        // Binäre Suche für effizientes Merging
        for (const item of newData) {
            const index = this._findInsertionPoint(item.timestamp);
            this.historicalData.splice(index, 0, item);
        }
    }

    _findInsertionPoint(timestamp) {
        let low = 0;
        let high = this.historicalData.length;

        while (low < high) {
            const mid = Math.floor((low + high) / 2);
            if (this.historicalData[mid].timestamp < timestamp) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        return low;
    }

    /**
     * Chunk Management
     */
    _initChunkManagement() {
        this._memoryCache = new Map();
        this._maxMemoryCacheSize = 50; // Maximum Chunks im Memory
        this._failedChunks = new Set();
        this._chunkLoadQueue = [];
        this._isLoadingChunk = false;

        // Retry-Logik für fehlgeschlagene Chunks
        setInterval(() => this._retryFailedChunks(), 30000);
    }

    async _retryFailedChunks() {
        for (const chunkId of this._failedChunks) {
            try {
                await this._loadChunk(chunkId);
                this._failedChunks.delete(chunkId);
            } catch (error) {
                console.warn(`Retry failed for chunk ${chunkId}:`, error);
            }
        }
    }

    /**
     * Datenbank Management
     */
    async _getDatabase() {
        if (this._db) return this._db;

        return new Promise((resolve, reject) => {
            const request = indexedDB.open('performanceMonitor', 1);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this._db = request.result;
                resolve(this._db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('chunks')) {
                    db.createObjectStore('chunks');
                }
            };
        });
    }

    /**
     * Optimierte Chart-Initialisierung
     */
    initCharts() {
        // Chart.js Performance-Optimierungen
        Chart.defaults.animation = false;  // Deaktiviere Animationen für bessere Performance
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;

        // Erweiterte Chart-Konfiguration
        const ctx = this.shadowRoot.querySelector('#performanceChart').getContext('2d');
        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: this._createInitialDatasets(),
            options: {
                parsing: false,  // Deaktiviere automatisches Parsing für bessere Performance
                normalized: true,  // Normalisierte Daten für schnelleres Rendering
                spanGaps: true,   // Verbinde Datenlücken
                
                elements: {
                    line: {
                        tension: 0.1,  // Reduzierte Linienkrümmung für bessere Performance
                    },
                    point: {
                        radius: 0,  // Keine Punkte für bessere Performance
                        hitRadius: 10,  // Aber noch interaktiv
                        hoverRadius: 5  // Zeige Punkte beim Hover
                    }
                },

                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute',
                            displayFormats: {
                                minute: 'HH:mm'
                            }
                        },
                        ticks: {
                            source: 'auto',
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        type: 'linear',
                        beginAtZero: true,
                        ticks: {
                            maxTicksLimit: 8
                        }
                    }
                },

                plugins: {
                    decimation: {
                        enabled: true,
                        algorithm: 'min-max',
                        threshold: 100
                    },
                    
                    tooltip: {
                        mode: 'nearest',
                        intersect: false,
                        position: 'nearest',
                        callbacks: {
                            label: this._formatTooltipLabel.bind(this)
                        }
                    },

                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x'
                        },
                        zoom: {
                            wheel: {
                                enabled: true
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x'
                        },
                        limits: {
                            x: {min: 'original', max: 'original'}
                        }
                    }
                },

                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });

        // Optimierte Event-Handler
        this.charts.performance.options.onHover = this._throttle(
            (event, elements) => this._handleChartHover(event, elements),
            100
        );
    }

    /**
     * Optimierte Datensatz-Erstellung
     */
    _createInitialDatasets() {
        return {
            datasets: this.metrics.map(metric => ({
                label: metric.name,
                data: [],
                borderColor: metric.color,
                backgroundColor: `${metric.color}33`,
                borderWidth: 1.5,
                fill: false,
                cubicInterpolationMode: 'monotone',
                pointHoverBackgroundColor: metric.color,
                parsing: false,  // Deaktiviere Parsing für bessere Performance
                normalized: true // Normalisierte Daten
            }))
        };
    }

    /**
     * Optimierte Chart-Updates
     */
    updateChart(newData) {
        if (!this.charts.performance) return;

        // Batch-Update für bessere Performance
        const batchSize = 1000;
        const chunks = this._chunkArray(newData, batchSize);

        requestAnimationFrame(() => {
            chunks.forEach((chunk, index) => {
                setTimeout(() => {
                    this._updateChartWithChunk(chunk);
                }, index * 16); // ~60fps
            });
        });
    }

    _updateChartWithChunk(chunk) {
        const chart = this.charts.performance;
        
        chunk.forEach(data => {
            chart.data.datasets.forEach(dataset => {
                if (dataset.label === data.metric) {
                    // Optimierte Datenpunkt-Hinzufügung
                    dataset.data.push({
                        x: data.timestamp,
                        y: data.value
                    });
                }
            });
        });

        // Entferne alte Datenpunkte
        const retention = 24 * 60 * 60 * 1000; // 24h
        const cutoff = Date.now() - retention;

        chart.data.datasets.forEach(dataset => {
            dataset.data = dataset.data.filter(point => point.x > cutoff);
        });

        // Optimiertes Update
        chart.update('quiet');
    }

    /**
     * Performance Utilities
     */
    _throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    _chunkArray(array, size) {
        const chunks = [];
        for (let i = 0; i < array.length; i += size) {
            chunks.push(array.slice(i, i + size));
        }
        return chunks;
    }

    /**
     * Heatmap Visualisierung
     */
    initHeatmap() {
        const ctx = this.shadowRoot.querySelector('#heatmapChart').getContext('2d');
        this.charts.heatmap = new Chart(ctx, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Performance Heatmap',
                    data: this._processHeatmapData(),
                    backgroundColor(context) {
                        const value = context.dataset.data[context.dataIndex].v;
                        const alpha = Math.min(Math.max(value / 100, 0), 1);
                        return `rgba(255, 99, 132, ${alpha})`;
                    },
                    width: ({ chart }) => (chart.chartArea.width / 24),
                    height: ({ chart }) => (chart.chartArea.height / 7)
                }]
            },
            options: {
                plugins: {
                    tooltip: {
                        callbacks: {
                            title() {
                                return '';
                            },
                            label(context) {
                                const v = context.dataset.data[context.dataIndex];
                                return [`Time: ${v.x}`, `Day: ${v.y}`, `Value: ${v.v}`];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        offset: true,
                        time: {
                            unit: 'hour',
                            displayFormats: {
                                hour: 'HH:mm'
                            }
                        },
                        ticks: {
                            maxRotation: 0
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        offset: true,
                        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    }
                }
            }
        });
    }

    /**
     * Waterfall Diagramm
     */
    initWaterfall() {
        const ctx = this.shadowRoot.querySelector('#waterfallChart').getContext('2d');
        this.charts.waterfall = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.waterfallData.map(d => d.label),
                datasets: [{
                    data: this.waterfallData.map(d => d.value),
                    backgroundColor: this.waterfallData.map(d => d.color),
                    borderColor: 'rgba(0, 0, 0, 0.1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const data = this.waterfallData[context.dataIndex];
                                return `${data.label}: ${data.value}ms`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Stack Trace Visualisierung
     */
    initStackTrace() {
        const container = this.shadowRoot.querySelector('#stackTraceContainer');
        const width = container.clientWidth;
        const height = 500;

        const tree = d3.tree().size([width - 100, height - 100]);
        const root = d3.hierarchy(this.stackTraceData);
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', 'translate(50,50)');

        const links = tree(root).links();
        const nodes = root.descendants();

        // Links zeichnen
        svg.selectAll('.link')
            .data(links)
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('d', d3.linkVertical()
                .x(d => d.x)
                .y(d => d.y));

        // Nodes zeichnen
        const node = svg.selectAll('.node')
            .data(nodes)
            .enter()
            .append('g')
            .attr('class', d => `node ${d.children ? 'node--internal' : 'node--leaf'}`)
            .attr('transform', d => `translate(${d.x},${d.y})`);

        node.append('circle')
            .attr('r', 5)
            .style('fill', d => d.data.error ? '#ff4444' : '#4CAF50');

        node.append('text')
            .attr('dy', '.31em')
            .attr('x', d => d.children ? -6 : 6)
            .style('text-anchor', d => d.children ? 'end' : 'start')
            .text(d => d.data.name);
    }

    /**
     * Datenverarbeitung für Visualisierungen
     */
    _processHeatmapData() {
        return this.historicalData.map(data => ({
            x: new Date(data.timestamp),
            y: new Date(data.timestamp).getDay(),
            v: data.value
        }));
    }

    _processWaterfallData() {
        return this.performanceMetrics.map((metric, index) => ({
            label: metric.name,
            value: metric.value,
            color: this._getWaterfallColor(index, metric.value)
        }));
    }

    _getWaterfallColor(index, value) {
        const colors = [
            'rgba(75, 192, 192, 0.7)',  // Türkis
            'rgba(255, 99, 132, 0.7)',   // Rot
            'rgba(54, 162, 235, 0.7)',   // Blau
            'rgba(255, 206, 86, 0.7)'    // Gelb
        ];
        return colors[index % colors.length];
    }

    _renderLineChartControls() {
        return html`
            <div class="control-panel">
                <div class="control-group">
                    <div class="control-group-title">Time Range</div>
                    <div class="control-row">
                        <select class="select-control" @change="${this._handleTimeRangeChange}">
                            <option value="1h">Last Hour</option>
                            <option value="24h" selected>Last 24 Hours</option>
                            <option value="7d">Last 7 Days</option>
                            <option value="30d">Last 30 Days</option>
                            <option value="custom">Custom Range</option>
                        </select>
                        ${this._renderCustomDateRange()}
                    </div>
                </div>

                <div class="control-group">
                    <div class="control-group-title">Metrics</div>
                    <div class="checkbox-group">
                        ${this.metrics.map(metric => html`
                            <label class="checkbox-label">
                                <input type="checkbox" 
                                       .checked="${metric.visible}"
                                       @change="${e => this._toggleMetric(metric.id, e.target.checked)}">
                                <span style="color: ${metric.color}">${metric.name}</span>
                            </label>
                        `)}
                    </div>
                </div>

                <div class="control-group">
                    <div class="control-group-title">Aggregation</div>
                    <select class="select-control" @change="${this._handleAggregationChange}">
                        <option value="none">No Aggregation</option>
                        <option value="avg">Average</option>
                        <option value="max">Maximum</option>
                        <option value="min">Minimum</option>
                    </select>
                </div>
            </div>
        `;
    }

    _renderHeatmapControls() {
        return html`
            <div class="control-panel">
                <div class="control-group">
                    <div class="control-group-title">Color Scale</div>
                    <div class="control-row">
                        <select class="select-control" @change="${this._handleColorScaleChange}">
                            <option value="sequential">Sequential</option>
                            <option value="diverging">Diverging</option>
                            <option value="custom">Custom</option>
                        </select>
                    </div>
                </div>

                <div class="control-group">
                    <div class="control-group-title">Grouping</div>
                    <select class="select-control" @change="${this._handleHeatmapGroupingChange}">
                        <option value="hour">By Hour</option>
                        <option value="day">By Day</option>
                        <option value="week">By Week</option>
                    </select>
                </div>

                <div class="control-group">
                    <div class="control-group-title">Threshold</div>
                    <div class="range-control">
                        <input type="range" 
                               min="0" 
                               max="100" 
                               .value="${this.heatmapThreshold}"
                               @input="${this._handleThresholdChange}">
                        <span class="range-value">${this.heatmapThreshold}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    _renderWaterfallControls() {
        return html`
            <div class="control-panel">
                <div class="control-group">
                    <div class="control-group-title">Request Filter</div>
                    <input type="text" 
                           class="select-control"
                           placeholder="Filter requests..."
                           @input="${this._handleRequestFilter}">
                </div>

                <div class="control-group">
                    <div class="control-group-title">Sort By</div>
                    <select class="select-control" @change="${this._handleWaterfallSort}">
                        <option value="time">Time</option>
                        <option value="duration">Duration</option>
                        <option value="size">Size</option>
                    </select>
                </div>

                <div class="control-group">
                    <div class="control-group-title">Detail Level</div>
                    <select class="select-control" @change="${this._handleDetailLevelChange}">
                        <option value="high">High</option>
                        <option value="medium">Medium</option>
                        <option value="low">Low</option>
                    </select>
                </div>
            </div>
        `;
    }

    _renderStackTraceControls() {
        return html`
            <div class="control-panel">
                <div class="control-group">
                    <div class="control-group-title">Zoom Level</div>
                    <div class="range-control">
                        <input type="range" 
                               min="0.5" 
                               max="2" 
                               step="0.1"
                               .value="${this.stackTraceZoom}"
                               @input="${this._handleZoomChange}">
                        <span class="range-value">${this.stackTraceZoom}x</span>
                    </div>
                </div>

                <div class="control-group">
                    <div class="control-group-title">Error Types</div>
                    <div class="checkbox-group">
                        ${this.errorTypes.map(type => html`
                            <label class="checkbox-label">
                                <input type="checkbox" 
                                       .checked="${type.visible}"
                                       @change="${e => this._toggleErrorType(type.id, e.target.checked)}">
                                ${type.name}
                            </label>
                        `)}
                    </div>
                </div>

                <div class="control-group">
                    <div class="control-group-title">Grouping</div>
                    <select class="select-control" @change="${this._handleStackTraceGrouping}">
                        <option value="none">No Grouping</option>
                        <option value="type">By Type</option>
                        <option value="module">By Module</option>
                    </select>
                </div>
            </div>
        `;
    }

    /**
     * Line Chart Event Handler
     */
    _handleTimeRangeChange(e) {
        const range = e.target.value;
        const now = Date.now();
        
        let start = now;
        switch(range) {
            case '1h':
                start = now - (60 * 60 * 1000);
                break;
            case '24h':
                start = now - (24 * 60 * 60 * 1000);
                break;
            case '7d':
                start = now - (7 * 24 * 60 * 60 * 1000);
                break;
            case '30d':
                start = now - (30 * 24 * 60 * 60 * 1000);
                break;
            case 'custom':
                // Custom range wird über separaten Dialog behandelt
                this._showCustomRangeDialog();
                return;
        }

        this.visibleTimeRange = { start, end: now };
        this._updateChartData();
    }

    _toggleMetric(metricId, visible) {
        const chart = this.charts.performance;
        const datasetIndex = chart.data.datasets.findIndex(
            ds => ds.label === metricId
        );

        if (datasetIndex !== -1) {
            chart.setDatasetVisibility(datasetIndex, visible);
            chart.update();
        }

        // Update metrics state
        this.metrics = this.metrics.map(metric => 
            metric.id === metricId ? {...metric, visible} : metric
        );
    }

    _handleAggregationChange(e) {
        const type = e.target.value;
        if (type === 'none') {
            this._updateChartData(this.rawData);
            return;
        }

        const aggregatedData = this._aggregateData(this.rawData, type);
        this._updateChartData(aggregatedData);
    }

    /**
     * Heatmap Event Handler
     */
    _handleColorScaleChange(e) {
        const scale = e.target.value;
        const heatmap = this.charts.heatmap;

        const colorScales = {
            sequential: (value) => `rgba(255, 99, 132, ${value})`,
            diverging: (value) => {
                const mid = 0.5;
                if (value < mid) {
                    return `rgba(66, 146, 198, ${value * 2})`;
                }
                return `rgba(239, 59, 44, ${(value - mid) * 2})`;
            },
            custom: (value) => this._customColorScale(value)
        };

        heatmap.options.plugins.colors.mapping = colorScales[scale];
        heatmap.update();
    }

    _handleHeatmapGroupingChange(e) {
        const grouping = e.target.value;
        const groupedData = this._groupHeatmapData(this.rawData, grouping);
        this._updateHeatmapData(groupedData);
    }

    _handleThresholdChange(e) {
        const threshold = parseInt(e.target.value);
        this.heatmapThreshold = threshold;
        this._updateHeatmapThresholds(threshold);
    }

    /**
     * Waterfall Event Handler
     */
    _handleRequestFilter(e) {
        const filterText = e.target.value.toLowerCase();
        const filteredData = this.waterfallData.filter(item =>
            item.label.toLowerCase().includes(filterText)
        );
        this._updateWaterfallChart(filteredData);
    }

    _handleWaterfallSort(e) {
        const sortType = e.target.value;
        const sortedData = [...this.waterfallData].sort((a, b) => {
            switch(sortType) {
                case 'time':
                    return a.startTime - b.startTime;
                case 'duration':
                    return b.duration - a.duration;
                case 'size':
                    return b.size - a.size;
                default:
                    return 0;
            }
        });
        this._updateWaterfallChart(sortedData);
    }

    _handleDetailLevelChange(e) {
        const level = e.target.value;
        this.detailLevel = level;
        this._updateWaterfallDetail(level);
    }

    /**
     * Stack Trace Event Handler
     */
    _handleZoomChange(e) {
        const zoom = parseFloat(e.target.value);
        this.stackTraceZoom = zoom;
        this._updateStackTraceZoom(zoom);
    }

    _toggleErrorType(typeId, visible) {
        this.errorTypes = this.errorTypes.map(type =>
            type.id === typeId ? {...type, visible} : type
        );
        this._updateStackTraceVisibility();
    }

    _handleStackTraceGrouping(e) {
        const grouping = e.target.value;
        const groupedData = this._groupStackTraceData(grouping);
        this._updateStackTraceVisualization(groupedData);
    }

    /**
     * Helper Methods für Datenverarbeitung
     */
    _aggregateData(data, type) {
        const aggregators = {
            avg: values => values.reduce((a, b) => a + b) / values.length,
            max: values => Math.max(...values),
            min: values => Math.min(...values)
        };

        return this._groupDataByTimeInterval(data).map(group => ({
            timestamp: group.timestamp,
            value: aggregators[type](group.values)
        }));
    }

    _groupDataByTimeInterval(data, interval = 3600000) { // 1h default
        const groups = {};
        data.forEach(item => {
            const timeGroup = Math.floor(item.timestamp / interval) * interval;
            if (!groups[timeGroup]) {
                groups[timeGroup] = { timestamp: timeGroup, values: [] };
            }
            groups[timeGroup].values.push(item.value);
        });
        return Object.values(groups);
    }

    _updateChartData(data = this.rawData) {
        if (!this.charts.performance) return;

        const chart = this.charts.performance;
        chart.data.datasets.forEach(dataset => {
            const metricData = data.filter(d => d.metric === dataset.label);
            dataset.data = metricData.map(d => ({
                x: d.timestamp,
                y: d.value
            }));
        });
        chart.update('none');
    }

    // ... rest of existing code ...

    /**
     * Custom Range Dialog
     */
    _showCustomRangeDialog() {
        const dialog = document.createElement('div');
        dialog.className = 'custom-range-dialog';
        dialog.innerHTML = `
            <div class="dialog-content">
                <h3>Custom Time Range</h3>
                
                <div class="quick-ranges">
                    <button data-range="last-week">Last Week</button>
                    <button data-range="last-month">Last Month</button>
                    <button data-range="custom">Custom Range</button>
                </div>

                <div class="date-inputs">
                    <div class="input-group">
                        <label>Start Date/Time</label>
                        <input type="datetime-local" id="start-time">
                    </div>
                    <div class="input-group">
                        <label>End Date/Time</label>
                        <input type="datetime-local" id="end-time">
                    </div>
                </div>

                <div class="preview-section">
                    <h4>Data Preview</h4>
                    <div class="preview-chart"></div>
                    <div class="data-points">
                        Available data points: <span id="point-count">0</span>
                    </div>
                </div>

                <div class="dialog-actions">
                    <button class="cancel-btn">Cancel</button>
                    <button class="apply-btn">Apply Range</button>
                </div>
            </div>
        `;

        // Styles für den Dialog
        const styles = document.createElement('style');
        styles.textContent = `
            .custom-range-dialog {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
            }

            .dialog-content {
                background: white;
                padding: 2rem;
                border-radius: 8px;
                min-width: 500px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }

            .quick-ranges {
                display: flex;
                gap: 1rem;
                margin-bottom: 1rem;
            }

            .date-inputs {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin-bottom: 1rem;
            }

            .preview-section {
                border-top: 1px solid #eee;
                padding-top: 1rem;
                margin-top: 1rem;
            }

            .preview-chart {
                height: 200px;
                margin: 1rem 0;
                background: #f5f5f5;
                border-radius: 4px;
            }

            .dialog-actions {
                display: flex;
                justify-content: flex-end;
                gap: 1rem;
                margin-top: 1rem;
            }
        `;

        dialog.appendChild(styles);
        document.body.appendChild(dialog);

        // Event Listener
        dialog.querySelector('.cancel-btn').addEventListener('click', () => {
            document.body.removeChild(dialog);
        });

        dialog.querySelector('.apply-btn').addEventListener('click', () => {
            const start = dialog.querySelector('#start-time').value;
            const end = dialog.querySelector('#end-time').value;
            this._applyCustomRange(new Date(start), new Date(end));
            document.body.removeChild(dialog);
        });

        // Quick Range Buttons
        dialog.querySelectorAll('.quick-ranges button').forEach(btn => {
            btn.addEventListener('click', () => {
                const range = btn.dataset.range;
                const now = new Date();
                let start = now;

                switch(range) {
                    case 'last-week':
                        start = new Date(now - 7 * 24 * 60 * 60 * 1000);
                        break;
                    case 'last-month':
                        start = new Date(now - 30 * 24 * 60 * 60 * 1000);
                        break;
                }

                dialog.querySelector('#start-time').value = this._formatDateTime(start);
                dialog.querySelector('#end-time').value = this._formatDateTime(now);
                this._updatePreview(start, now);
            });
        });

        // Initial Preview
        this._updatePreview(new Date(), new Date());
    }

    _updatePreview(start, end) {
        const previewData = this._getDataInRange(start, end);
        const pointCount = previewData.length;
        document.querySelector('#point-count').textContent = pointCount;

        // Mini-Chart Preview
        this._renderPreviewChart(previewData);
    }

    _formatDateTime(date) {
        return date.toISOString().slice(0, 16);
    }

    /**
     * Preview Chart Implementation
     */
    _renderPreviewChart(data) {
        const container = document.querySelector('.preview-chart');
        if (!container) return;

        // Cleanup existing chart
        if (this.previewChart) {
            this.previewChart.destroy();
        }

        // Daten für Preview optimieren
        const optimizedData = this._optimizePreviewData(data);

        // Preview Chart Konfiguration
        this.previewChart = new Chart(container.getContext('2d'), {
            type: 'line',
            data: {
                datasets: this.metrics.map(metric => ({
                    label: metric.name,
                    data: optimizedData.filter(d => d.metric === metric.id),
                    borderColor: metric.color,
                    backgroundColor: `${metric.color}33`,
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour',
                            displayFormats: {
                                hour: 'HH:mm'
                            }
                        },
                        ticks: {
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 5
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            maxTicksLimit: 3
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: true,
                        mode: 'index',
                        intersect: false,
                        position: 'nearest',
                        callbacks: {
                            label: (context) => {
                                const value = context.parsed.y;
                                const metric = this.metrics[context.datasetIndex];
                                return `${metric.name}: ${this._formatValue(value, metric.unit)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Datenoptimierung für Preview
     */
    _optimizePreviewData(data) {
        // Maximale Anzahl von Datenpunkten für die Vorschau
        const MAX_PREVIEW_POINTS = 100;
        
        if (data.length <= MAX_PREVIEW_POINTS) {
            return data;
        }

        // Sampling-Rate berechnen
        const samplingRate = Math.ceil(data.length / MAX_PREVIEW_POINTS);
        
        // Daten gruppieren und aggregieren
        const groupedData = {};
        data.forEach((point, index) => {
            const groupIndex = Math.floor(index / samplingRate);
            if (!groupedData[groupIndex]) {
                groupedData[groupIndex] = {
                    sum: 0,
                    count: 0,
                    metric: point.metric,
                    timestamp: point.timestamp
                };
            }
            groupedData[groupIndex].sum += point.value;
            groupedData[groupIndex].count++;
        });

        // Durchschnitte berechnen
        return Object.values(groupedData).map(group => ({
            metric: group.metric,
            timestamp: group.timestamp,
            value: group.sum / group.count
        }));
    }

    /**
     * Wert-Formatierung
     */
    _formatValue(value, unit) {
        if (typeof value !== 'number') return 'N/A';

        // Einheiten-spezifische Formatierung
        switch (unit) {
            case 'ms':
                return `${value.toFixed(2)}ms`;
            case 'bytes':
                return this._formatBytes(value);
            case 'percent':
                return `${value.toFixed(1)}%`;
            default:
                return value.toFixed(2);
        }
    }

    /**
     * Bytes-Formatierung
     */
    _formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
    }

    /**
     * Event Handler für Preview Interaktionen
     */
    _initPreviewInteractions() {
        const container = document.querySelector('.preview-chart');
        if (!container) return;

        // Zoom-Handler
        let isDragging = false;
        let startX = 0;
        let startRange = null;

        container.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.offsetX;
            startRange = {
                start: this.previewChart.scales.x.min,
                end: this.previewChart.scales.x.max
            };
        });

        container.addEventListener('mousemove', (e) => {
            if (!isDragging || !startRange) return;

            const deltaX = e.offsetX - startX;
            const rangeWidth = startRange.end - startRange.start;
            const moveAmount = (deltaX / container.offsetWidth) * rangeWidth;

            const newStart = new Date(startRange.start - moveAmount);
            const newEnd = new Date(startRange.end - moveAmount);

            this._updateDateInputs(newStart, newEnd);
            this._updatePreview(newStart, newEnd);
        });

        container.addEventListener('mouseup', () => {
            isDragging = false;
            startRange = null;
        });

        container.addEventListener('mouseleave', () => {
            isDragging = false;
            startRange = null;
        });
    }

    /**
     * Chart Synchronisation
     */
    _initChartSync() {
        // Gemeinsamer State für Zeitbereiche
        this.timeRangeState = {
            start: null,
            end: null,
            syncInProgress: false
        };

        // Event Listener für Hauptchart
        this.charts.performance.options.plugins.zoom.zoom.onZoom = (event) => {
            if (this.timeRangeState.syncInProgress) return;
            this.timeRangeState.syncInProgress = true;

            const { min, max } = event.chart.scales.x;
            this._syncTimeRange(min, max);
        };

        this.charts.performance.options.plugins.zoom.pan.onPan = (event) => {
            if (this.timeRangeState.syncInProgress) return;
            this.timeRangeState.syncInProgress = true;

            const { min, max } = event.chart.scales.x;
            this._syncTimeRange(min, max);
        };
    }

    /**
     * Synchronisiere Zeitbereiche zwischen Charts
     */
    _syncTimeRange(start, end) {
        try {
            // Update gemeinsamen State
            this.timeRangeState.start = start;
            this.timeRangeState.end = end;

            // Update Hauptchart
            this._updateMainChartRange(start, end);

            // Update Preview wenn vorhanden
            if (this.previewChart) {
                this._updatePreviewChartRange(start, end);
            }

            // Update Custom Range Dialog wenn offen
            this._updateCustomRangeInputs(start, end);

            // Dispatch Event für andere interessierte Komponenten
            this.dispatchEvent(new CustomEvent('timerange-changed', {
                detail: { start, end }
            }));
        } finally {
            // Reset sync flag
            setTimeout(() => {
                this.timeRangeState.syncInProgress = false;
            }, 100);
        }
    }

    /**
     * Update Hauptchart Bereich
     */
    _updateMainChartRange(start, end) {
        const chart = this.charts.performance;
        
        // Prüfe ob Update nötig
        if (chart.scales.x.min === start && chart.scales.x.max === end) {
            return;
        }

        // Update Achsen
        chart.scales.x.options.min = start;
        chart.scales.x.options.max = end;

        // Lade Daten für neuen Bereich
        this._loadDataForRange(start, end).then(data => {
            this._updateChartData(chart, data);
        });

        // Update ohne Animation
        chart.update('none');
    }

    /**
     * Update Preview Chart Bereich
     */
    _updatePreviewChartRange(start, end) {
        if (!this.previewChart) return;

        // Update Achsen
        this.previewChart.scales.x.options.min = start;
        this.previewChart.options.x.max = end;

        // Optimierte Daten für Preview
        const previewData = this._optimizePreviewData(
            this._getDataInRange(start, end)
        );

        // Update Datasets
        this.previewChart.data.datasets.forEach((dataset, index) => {
            dataset.data = previewData.filter(d => 
                d.metric === this.metrics[index].id
            );
        });

        // Update ohne Animation
        this.previewChart.update('none');
    }

    /**
     * Update Dialog Inputs
     */
    _updateCustomRangeInputs(start, end) {
        const startInput = document.querySelector('#start-time');
        const endInput = document.querySelector('#end-time');

        if (startInput && endInput) {
            startInput.value = this._formatDateTime(new Date(start));
            endInput.value = this._formatDateTime(new Date(end));
        }
    }

    /**
     * Daten für Zeitbereich laden
     */
    async _loadDataForRange(start, end) {
        // Prüfe Cache
        const cachedData = this._getDataFromCache(start, end);
        if (cachedData) {
            return cachedData;
        }

        // Berechne benötigte Chunks
        const chunks = this._calculateRequiredChunks(start, end);
        
        // Lade fehlende Chunks
        const loadPromises = chunks
            .filter(chunk => !this.loadedChunks.includes(chunk))
            .map(chunk => this._loadChunk(chunk));

        await Promise.all(loadPromises);

        // Filtere und return Daten für Bereich
        return this._getDataInRange(start, end);
    }

    /**
     * Cache Management
     */
    _getDataFromCache(start, end) {
        // Implementierung des Cache-Lookups
        if (!this._db) return null;

        return new Promise((resolve) => {
            const transaction = this._db.transaction(['chunks'], 'readonly');
            const store = transaction.objectStore('chunks');
            const range = IDBKeyRange.bound(start, end);

            const data = [];
            store.openCursor(range).onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    data.push(...cursor.value);
                    cursor.continue();
                } else {
                    resolve(data);
                }
            };
        });
    }

    /**
     * Flame Graph Visualisierung
     */
    initFlameGraph() {
        const container = this.shadowRoot.querySelector('#flameGraphChart');
        const width = container.clientWidth;
        const height = 400;

        // D3.js Flame Graph Setup
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Flame Graph Layout
        const flameGraph = d3.flamegraph()
            .width(width)
            .height(height)
            .cellHeight(18)
            .transitionDuration(750)
            .minFrameSize(5)
            .tooltip(true)
            .inverted(false)
            .sort(true);

        // Farb-Schema
        const colorMapper = d3.scaleOrdinal(d3.schemeCategory10);
        flameGraph.setColorMapper((d) => {
            return d.highlight ? '#E73F3F' : colorMapper(d.name);
        });

        // Daten rendern
        svg.datum(this.flameData)
            .call(flameGraph);

        // Interaktivität
        this._setupFlameGraphInteractions(flameGraph, svg);
    }

    _setupFlameGraphInteractions(flameGraph, svg) {
        // Zoom Controls
        const zoom = d3.zoom()
            .scaleExtent([0.5, 2])
            .on('zoom', (event) => {
                svg.attr('transform', event.transform);
            });

        svg.call(zoom);

        // Click Handler
        flameGraph.onClick((d) => {
            // Frame Details anzeigen
            this._showFrameDetails(d);
        });

        // Tooltip Anpassung
        flameGraph.tooltip((d) => {
            return `
                <div class="flame-tooltip">
                    <strong>${d.name}</strong>
                    <div>Zeit: ${d.value}ms</div>
                    <div>Samples: ${d.samples}</div>
                </div>
            `;
        });
    }

    _showFrameDetails(frame) {
        const details = this.shadowRoot.querySelector('#frameDetails');
        details.innerHTML = `
            <div class="frame-details">
                <h3>${frame.name}</h3>
                <table>
                    <tr>
                        <td>Ausführungszeit:</td>
                        <td>${frame.value}ms</td>
                    </tr>
                    <tr>
                        <td>Samples:</td>
                        <td>${frame.samples}</td>
                    </tr>
                    <tr>
                        <td>Anteil:</td>
                        <td>${(frame.value / this.totalTime * 100).toFixed(2)}%</td>
                    </tr>
                </table>
                ${this._renderChildFrames(frame)}
            </div>
        `;
    }

    _renderChildFrames(frame) {
        if (!frame.children || frame.children.length === 0) return '';

        return `
            <div class="child-frames">
                <h4>Child Frames</h4>
                <ul>
                    ${frame.children.map(child => `
                        <li>
                            ${child.name} (${child.value}ms)
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }

    /**
     * Correlation Plot Visualisierung
     * Zeigt wie verschiedene Metriken miteinander zusammenhängen
     * z.B.: Steigt die Response-Zeit wenn die Server-Last steigt?
     */
    initCorrelationPlot() {
        const container = this.shadowRoot.querySelector('#correlationPlotContainer');
        const width = container.clientWidth;
        const height = 400;

        // Erstelle Basis SVG
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Definiere Ränder
        const margin = {top: 40, right: 40, bottom: 60, left: 60};
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        // Erstelle Scales für X und Y Achsen
        const xScale = d3.scaleLinear()
            .domain([0, d3.max(this.correlationData, d => d.x)])
            .range([0, plotWidth]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(this.correlationData, d => d.y)])
            .range([plotHeight, 0]);

        // Füge Achsen hinzu
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        g.append('g')
            .attr('transform', `translate(0,${plotHeight})`)
            .call(d3.axisBottom(xScale))
            .append('text')
            .attr('x', plotWidth / 2)
            .attr('y', 40)
            .attr('fill', 'currentColor')
            .text(this.xAxisMetric);

        g.append('g')
            .call(d3.axisLeft(yScale))
            .append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -40)
            .attr('x', -plotHeight / 2)
            .attr('fill', 'currentColor')
            .text(this.yAxisMetric);

        // Füge Datenpunkte hinzu
        g.selectAll('circle')
            .data(this.correlationData)
            .enter()
            .append('circle')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 5)
            .attr('fill', 'steelblue')
            .attr('opacity', 0.6);

        // Füge Trendlinie hinzu
        if (this.correlationData.length > 1) {
            const lineGenerator = d3.line()
                .x(d => xScale(d.x))
                .y(d => yScale(d.y));

            const regression = this._calculateRegression(this.correlationData);
            const lineData = [
                {x: d3.min(this.correlationData, d => d.x), y: regression.predict(d3.min(this.correlationData, d => d.x))},
                {x: d3.max(this.correlationData, d => d.x), y: regression.predict(d3.max(this.correlationData, d => d.x))}
            ];

            g.append('path')
                .datum(lineData)
                .attr('fill', 'none')
                .attr('stroke', 'red')
                .attr('stroke-width', 1.5)
                .attr('d', lineGenerator);

            // Zeige Korrelationskoeffizient
            g.append('text')
                .attr('x', 10)
                .attr('y', 20)
                .text(`Korrelation: ${regression.r2.toFixed(3)}`);
        }

        // Interaktive Features
        this._addCorrelationInteractions(g, xScale, yScale);
    }

    /**
     * Berechnet die lineare Regression für die Datenpunkte
     * Hilft zu verstehen, wie stark zwei Metriken korrelieren
     */
    _calculateRegression(data) {
        const xMean = d3.mean(data, d => d.x);
        const yMean = d3.mean(data, d => d.y);
        
        const ssxx = d3.sum(data, d => Math.pow(d.x - xMean, 2));
        const ssyy = d3.sum(data, d => Math.pow(d.y - yMean, 2));
        const ssxy = d3.sum(data, d => (d.x - xMean) * (d.y - yMean));
        
        const slope = ssxy / ssxx;
        const intercept = yMean - slope * xMean;
        const r2 = Math.pow(ssxy, 2) / (ssxx * ssyy);

        return {
            predict: x => slope * x + intercept,
            r2: r2
        };
    }

    /**
     * Fügt interaktive Features zum Plot hinzu
     * - Tooltips bei Hover
     * - Zoom für detailliertere Analyse
     * - Highlighting von Datenpunkten
     */
    _addCorrelationInteractions(g, xScale, yScale) {
        // Tooltip
        const tooltip = d3.select(this.shadowRoot)
            .append('div')
            .attr('class', 'correlation-tooltip')
            .style('opacity', 0);

        g.selectAll('circle')
            .on('mouseover', (event, d) => {
                tooltip.transition()
                    .duration(200)
                    .style('opacity', .9);
                tooltip.html(`
                    ${this.xAxisMetric}: ${d.x}<br/>
                    ${this.yAxisMetric}: ${d.y}
                `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', () => {
                tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
            });

        // Zoom Funktionalität
        const zoom = d3.zoom()
            .scaleExtent([0.5, 5])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        g.call(zoom);
    }

    /**
     * System Resource Map
     * Visualisiert Systemressourcen als interaktive Treemap
     * Größe = Ressourcennutzung
     * Farbe = Auslastungsgrad
     */
    initSystemResourceMap() {
        const container = this.shadowRoot.querySelector('#resourceMapContainer');
        const width = container.clientWidth;
        const height = 500;

        // Basis SVG erstellen
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Treemap Layout
        const treemap = d3.treemap()
            .size([width, height])
            .padding(1)
            .round(true);

        // Farbskala für Auslastung (grün -> gelb -> rot)
        const colorScale = d3.scaleLinear()
            .domain([0, 50, 100])
            .range(['#00ff00', '#ffff00', '#ff0000']);

        // Hierarchische Datenstruktur erstellen
        const root = d3.hierarchy({
            name: 'resources',
            children: this.resourceData
        })
            .sum(d => d.value)
            .sort((a, b) => b.value - a.value);

        // Treemap Layout anwenden
        treemap(root);

        // Ressourcen-Zellen zeichnen
        const cell = svg.selectAll('g')
            .data(root.leaves())
            .enter().append('g')
            .attr('transform', d => `translate(${d.x0},${d.y0})`);

        // Rechtecke für jede Ressource
        cell.append('rect')
            .attr('width', d => d.x1 - d.x0)
            .attr('height', d => d.y1 - d.y0)
            .attr('fill', d => colorScale(d.data.usage))
            .attr('stroke', '#fff')
            .attr('class', 'resource-cell')
            .on('mouseover', this._showResourceTooltip.bind(this))
            .on('mouseout', this._hideResourceTooltip.bind(this));

        // Labels für Ressourcen
        cell.append('text')
            .attr('x', 5)
            .attr('y', 15)
            .text(d => this._truncateText(d.data.name, d.x1 - d.x0))
            .attr('font-size', '12px')
            .attr('fill', d => d.data.usage > 70 ? '#fff' : '#000');

        // Usage-Wert
        cell.append('text')
            .attr('x', 5)
            .attr('y', 30)
            .text(d => `${d.data.usage}%`)
            .attr('font-size', '10px')
            .attr('fill', d => d.data.usage > 70 ? '#fff' : '#000');

        // Legende hinzufügen
        this._addResourceMapLegend(svg, width, colorScale);
    }

    /**
     * Tooltip für detaillierte Ressourcen-Informationen
     */
    _showResourceTooltip(event, d) {
        const tooltip = d3.select(this.shadowRoot.querySelector('.resource-tooltip'));
        
        tooltip.style('opacity', 1)
            .html(`
                <div class="tooltip-header">${d.data.name}</div>
                <div class="tooltip-content">
                    <div>Auslastung: ${d.data.usage}%</div>
                    <div>Absolut: ${d.data.value} ${d.data.unit}</div>
                    ${d.data.details ? `<div>${d.data.details}</div>` : ''}
                </div>
            `)
            .style('left', `${event.pageX + 10}px`)
            .style('top', `${event.pageY - 28}px`);
    }

    _hideResourceTooltip() {
        d3.select(this.shadowRoot.querySelector('.resource-tooltip'))
            .style('opacity', 0);
    }

    /**
     * Legende für die Farbskala
     */
    _addResourceMapLegend(svg, width, colorScale) {
        const legendWidth = 200;
        const legendHeight = 15;
        const margin = 20;

        const legend = svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${width - legendWidth - margin},${margin})`);

        // Gradient für die Legende
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'resource-gradient')
            .attr('x1', '0%')
            .attr('x2', '100%');

        gradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', colorScale(0));

        gradient.append('stop')
            .attr('offset', '50%')
            .attr('stop-color', colorScale(50));

        gradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', colorScale(100));

        // Legende Rechteck
        legend.append('rect')
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('fill', 'url(#resource-gradient)');

        // Legende Labels
        const labels = ['0%', '50%', '100%'];
        const labelPositions = [0, legendWidth/2, legendWidth];

        legend.selectAll('.legend-label')
            .data(labels)
            .enter()
            .append('text')
            .attr('class', 'legend-label')
            .attr('x', (d, i) => labelPositions[i])
            .attr('y', legendHeight + 15)
            .attr('text-anchor', (d, i) => i === 1 ? 'middle' : (i === 0 ? 'start' : 'end'))
            .text(d => d)
            .attr('font-size', '10px');
    }

    /**
     * Text kürzen wenn zu lang für Zelle
     */
    _truncateText(text, width) {
        if (text.length * 6 > width) { // Grobe Schätzung der Textbreite
            return text.slice(0, Math.floor(width/6)) + '...';
        }
        return text;
    }

    _updateXAxis(e) {
        this.xAxisMetric = e.target.value;
        this.initCorrelationPlot();
    }

    _updateYAxis(e) {
        this.yAxisMetric = e.target.value;
        this.initCorrelationPlot();
    }
}

customElements.define('bruno-performance-monitor', PerformanceMonitor);

export default PerformanceMonitor; 