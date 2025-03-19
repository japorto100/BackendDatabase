import { compress } from 'pako'; // Für GZIP Kompression
import { quantize } from './utils/data-processing.js';

/**
 * Web Worker für Datenkompression
 */
self.onmessage = async function(e) {
    const { type, data } = e.data;
    
    if (type === 'compress') {
        const compressed = await compressData(data);
        self.postMessage(compressed);
    }
};

async function compressData(data) {
    // Implementiere verschiedene Kompressionsstrategien
    const strategies = [
        downsampleTimeSeries,
        removeRedundantData,
        quantizeValues
    ];

    let result = data;
    for (const strategy of strategies) {
        result = await strategy(result);
    }

    return result;
}

function downsampleTimeSeries(data, targetPoints = 1000) {
    if (data.length <= targetPoints) return data;
    
    const factor = Math.ceil(data.length / targetPoints);
    return data.filter((_, i) => i % factor === 0);
}

function removeRedundantData(data) {
    return data.filter((item, index, array) => {
        if (index === 0) return true;
        
        // Vergleiche mit vorherigem Datenpunkt
        const prev = array[index - 1];
        return Object.keys(item).some(key => {
            if (key === 'timestamp') return true;
            return Math.abs(item[key] - prev[key]) > 0.001;
        });
    });
}

function quantizeValues(data, precision = 2) {
    return data.map(item => {
        const result = { ...item };
        for (const [key, value] of Object.entries(item)) {
            if (typeof value === 'number' && key !== 'timestamp') {
                result[key] = Number(value.toFixed(precision));
            }
        }
        return result;
    });
} 