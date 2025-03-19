/**
 * Alert Templates für verschiedene Performance-Szenarien
 */
export const AlertTemplates = {
    // Performance Templates
    RESPONSE_TIME: {
        id: 'response_time',
        name: 'Response Time Alert',
        description: 'Monitors API response times for degradation',
        metric: 'avgResponseTime',
        conditions: {
            warning: {
                threshold: 500,
                duration: '5m'
            },
            critical: {
                threshold: 1000,
                duration: '1m'
            }
        },
        suggestions: [
            'Check for database query optimization',
            'Verify cache hit rates',
            'Monitor server resource usage',
            'Check for network latency issues'
        ]
    },

    MEMORY_USAGE: {
        id: 'memory_usage',
        name: 'Memory Usage Alert',
        description: 'Monitors application memory consumption',
        metric: 'memoryUsage',
        conditions: {
            warning: {
                threshold: 70, // 70% usage
                duration: '10m'
            },
            critical: {
                threshold: 85, // 85% usage
                duration: '5m'
            }
        },
        suggestions: [
            'Check for memory leaks',
            'Review garbage collection metrics',
            'Consider scaling resources',
            'Analyze heap usage patterns'
        ]
    },

    ERROR_RATE: {
        id: 'error_rate',
        name: 'Error Rate Spike',
        description: 'Monitors application error frequency',
        metric: 'errorRate',
        conditions: {
            warning: {
                threshold: 1, // 1% error rate
                duration: '5m'
            },
            critical: {
                threshold: 5, // 5% error rate
                duration: '1m'
            }
        },
        suggestions: [
            'Check error logs for patterns',
            'Monitor external service dependencies',
            'Review recent code deployments',
            'Analyze affected endpoints'
        ]
    },

    CACHE_PERFORMANCE: {
        id: 'cache_performance',
        name: 'Cache Performance Alert',
        description: 'Monitors cache hit rates',
        metric: 'cacheHitRate',
        conditions: {
            warning: {
                threshold: 60, // Below 60% hit rate
                duration: '15m'
            },
            critical: {
                threshold: 40, // Below 40% hit rate
                duration: '5m'
            }
        },
        suggestions: [
            'Review cache invalidation strategy',
            'Check cache size configuration',
            'Analyze cache key patterns',
            'Monitor cache eviction rates'
        ]
    }
};

/**
 * Template Manager für Alert-Konfigurationen
 */
export class AlertTemplateManager {
    constructor() {
        this.templates = AlertTemplates;
        this.customTemplates = new Map();
    }

    getTemplate(id) {
        return this.templates[id] || this.customTemplates.get(id);
    }

    addCustomTemplate(template) {
        if (!template.id) {
            throw new Error('Template must have an ID');
        }
        this.customTemplates.set(template.id, template);
    }

    removeCustomTemplate(id) {
        this.customTemplates.delete(id);
    }

    getAllTemplates() {
        return {
            ...this.templates,
            ...Object.fromEntries(this.customTemplates)
        };
    }

    applyTemplate(id, customThresholds = {}) {
        const template = this.getTemplate(id);
        if (!template) {
            throw new Error(`Template ${id} not found`);
        }

        return {
            ...template,
            conditions: {
                warning: {
                    ...template.conditions.warning,
                    ...customThresholds.warning
                },
                critical: {
                    ...template.conditions.critical,
                    ...customThresholds.critical
                }
            }
        };
    }
} 