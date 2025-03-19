/**
 * Performance Monitor Web Worker
 * Handles data processing and aggregation in a separate thread
 */

self.onmessage = function(e) {
    const { type, data } = e.data;
    
    switch (type) {
        case 'processData':
            const aggregatedData = aggregateData(
                data.historicalData,
                data.resolution,
                data.maxPoints,
                data.selectedMetric
            );
            self.postMessage({ type: 'aggregatedData', data: aggregatedData });
            break;
    }
};

function aggregateData(data, resolution, maxPoints, metric) {
    // Gruppiere Daten nach Zeitintervall
    const interval = getIntervalMs(resolution);
    const groups = new Map();

    data.forEach(entry => {
        const timestamp = new Date(entry.timestamp).getTime();
        const bucket = Math.floor(timestamp / interval) * interval;
        
        if (!groups.has(bucket)) {
            groups.set(bucket, []);
        }
        groups.get(bucket).push(entry[metric]);
    });

    // Aggregiere Daten pro Gruppe
    const aggregated = Array.from(groups.entries()).map(([timestamp, values]) => ({
        timestamp,
        value: calculateAggregates(values)
    }));

    // Sortiere nach Zeit und beschränke auf maxPoints
    return aggregated
        .sort((a, b) => a.timestamp - b.timestamp)
        .slice(-maxPoints);
}

function getIntervalMs(resolution) {
    const intervals = {
        '1s': 1000,
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000
    };
    return intervals[resolution] || intervals['1m'];
}

function calculateAggregates(values) {
    if (!values.length) return null;

    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / values.length;
    const sorted = [...values].sort((a, b) => a - b);
    const median = sorted[Math.floor(values.length / 2)];
    const min = sorted[0];
    const max = sorted[values.length - 1];

    return {
        avg,
        median,
        min,
        max,
        count: values.length
    };
} 