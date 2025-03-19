import { saveAs } from 'file-saver';
import * as XLSX from 'xlsx';
import { jsPDF } from 'jspdf';

/**
 * Export Manager für Performance Daten
 * Handhabt verschiedene Export-Formate und Daten-Optimierung
 */
export class ExportManager {
    constructor(workerPath = '/static/js/bruno/compression-worker.js') {
        try {
            this.compressionWorker = new Worker(new URL(workerPath, import.meta.url));
            this.compressionWorker.onerror = (error) => {
                console.error('Compression Worker Error:', error);
                this._fallbackCompression = true;
            };
        } catch (error) {
            console.warn('Worker initialization failed, using fallback compression:', error);
            this._fallbackCompression = true;
        }
        
        this.maxExportSize = 50 * 1024 * 1024; // 50MB limit
    }

    /**
     * Exportiert Daten in verschiedenen Formaten
     */
    async exportData(data, format, options = {}) {
        const processedData = await this._preprocessData(data, options);
        
        switch (format) {
            case 'json':
                return this._exportJSON(processedData, options);
            case 'csv':
                return this._exportCSV(processedData, options);
            case 'excel':
                return this._exportExcel(processedData, options);
            case 'pdf':
                return this._exportPDF(processedData, options);
            default:
                throw new Error(`Unsupported format: ${format}`);
        }
    }

    /**
     * Vorverarbeitung und Optimierung der Daten
     */
    async _preprocessData(data, options) {
        const { timeRange, metrics, resolution } = options;
        
        // Filtere nach Zeitbereich
        let filteredData = this._filterByTimeRange(data, timeRange);
        
        // Aggregiere Daten basierend auf Auflösung
        filteredData = this._aggregateData(filteredData, resolution);
        
        // Komprimiere große Datensätze
        if (this._estimateSize(filteredData) > this.maxExportSize) {
            filteredData = await this._compressData(filteredData);
        }

        return filteredData;
    }

    /**
     * Komprimierung mit Worker oder Fallback
     */
    async _compressData(data) {
        if (this._fallbackCompression) {
            return this._fallbackCompress(data);
        }

        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error('Compression worker timeout'));
                this._fallbackCompression = true;
            }, 30000); // 30s timeout

            this.compressionWorker.onmessage = (e) => {
                clearTimeout(timeoutId);
                resolve(e.data);
            };

            this.compressionWorker.onerror = (e) => {
                clearTimeout(timeoutId);
                reject(e);
                this._fallbackCompression = true;
            };

            this.compressionWorker.postMessage({ type: 'compress', data });
        });
    }

    /**
     * Fallback Komprimierung wenn Worker nicht verfügbar
     */
    _fallbackCompress(data) {
        return {
            ...data,
            metrics: this._downsampleData(data.metrics)
        };
    }

    _downsampleData(data, targetPoints = 1000) {
        if (data.length <= targetPoints) return data;
        
        const factor = Math.ceil(data.length / targetPoints);
        return data.filter((_, i) => i % factor === 0);
    }

    /**
     * JSON Export mit Optimierungen
     */
    _exportJSON(data, options) {
        const { pretty = false } = options;
        const json = pretty 
            ? JSON.stringify(data, null, 2)
            : JSON.stringify(data);
        
        return this._downloadFile(json, 'performance-data.json', 'application/json');
    }

    /**
     * CSV Export mit Chunk-Processing
     */
    _exportCSV(data, options) {
        const { delimiter = ',', includeHeaders = true } = options;
        const chunks = this._chunkData(data, 1000); // Process 1000 rows at a time
        
        let csv = '';
        if (includeHeaders) {
            csv = Object.keys(data[0]).join(delimiter) + '\n';
        }

        for (const chunk of chunks) {
            csv += chunk.map(row => 
                Object.values(row).join(delimiter)
            ).join('\n');
        }

        return this._downloadFile(csv, 'performance-data.csv', 'text/csv');
    }

    /**
     * Excel Export mit Formatierung
     */
    async _exportExcel(data, options) {
        const XLSX = await import('xlsx');
        const wb = XLSX.utils.book_new();
        const ws = XLSX.utils.json_to_sheet(data);

        // Formatierung
        ws['!cols'] = this._getColumnWidths(data);
        
        XLSX.utils.book_append_sheet(wb, ws, 'Performance Data');
        XLSX.writeFile(wb, 'performance-data.xlsx');
    }

    /**
     * PDF Export mit Charts
     */
    async _exportPDF(data, options) {
        const { jsPDF } = await import('jspdf');
        const doc = new jsPDF();

        // Füge Metadaten hinzu
        doc.setProperties({
            title: 'Performance Report',
            subject: 'API Performance Metrics',
            author: 'Bruno Performance Monitor',
            keywords: 'performance, metrics, monitoring',
            creator: 'Export Manager'
        });

        // Füge Inhalt hinzu
        this._addPDFContent(doc, data, options);

        return doc.save('performance-report.pdf');
    }

    /**
     * Hilfsmethoden
     */
    _filterByTimeRange(data, timeRange) {
        if (!timeRange) return data;
        
        const now = Date.now();
        const startTime = now - this._parseTimeRange(timeRange);
        
        return data.filter(item => 
            new Date(item.timestamp).getTime() >= startTime
        );
    }

    _parseTimeRange(timeRange) {
        const units = {
            m: 60 * 1000,
            h: 60 * 60 * 1000,
            d: 24 * 60 * 60 * 1000
        };
        const match = timeRange.match(/^(\d+)([mhd])$/);
        if (!match) throw new Error('Invalid time range format');
        
        return match[1] * units[match[2]];
    }

    _aggregateData(data, resolution) {
        if (!resolution) return data;
        
        const interval = this._parseTimeRange(resolution);
        const aggregated = new Map();

        for (const item of data) {
            const bucket = Math.floor(new Date(item.timestamp).getTime() / interval) * interval;
            if (!aggregated.has(bucket)) {
                aggregated.set(bucket, []);
            }
            aggregated.get(bucket).push(item);
        }

        return Array.from(aggregated.entries()).map(([timestamp, items]) => ({
            timestamp: new Date(timestamp),
            ...this._calculateAggregates(items)
        }));
    }

    _calculateAggregates(items) {
        const metrics = Object.keys(items[0]).filter(key => key !== 'timestamp');
        const result = {};

        for (const metric of metrics) {
            const values = items.map(item => item[metric]).filter(v => v != null);
            result[metric] = {
                avg: values.reduce((a, b) => a + b, 0) / values.length,
                min: Math.min(...values),
                max: Math.max(...values)
            };
        }

        return result;
    }

    _chunkData(data, size) {
        const chunks = [];
        for (let i = 0; i < data.length; i += size) {
            chunks.push(data.slice(i, i + size));
        }
        return chunks;
    }

    _estimateSize(data) {
        return new Blob([JSON.stringify(data)]).size;
    }

    _downloadFile(content, filename, type) {
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Cleanup
     */
    destroy() {
        if (this.compressionWorker) {
            this.compressionWorker.terminate();
        }
    }
} 