/**
 * Vordefinierte Provider-Vorlagen für häufig verwendete Dienste
 */
export const PROVIDER_TEMPLATES = {
    // API-basierte Dienste
    'apollo': {
        name: 'Apollo.io',
        icon: '🚀',
        providerType: 'api',
        baseUrl: 'https://api.apollo.io/v1',
        description: 'B2B Sales Intelligence & Engagement Platform',
        requiresApiKey: true,
        defaultConfig: {
            endpoints: {
                search: '/mixed_people/search',
                enrich: '/people/enrich'
            }
        }
    },
    'github': {
        name: 'GitHub',
        icon: '🐙',
        providerType: 'api',
        baseUrl: 'https://api.github.com',
        description: 'Code-Hosting-Plattform',
        requiresApiKey: true,
        defaultConfig: {
            endpoints: {
                search: '/search/repositories',
                user: '/user'
            }
        }
    },
    'zefix': {
        name: 'Zefix',
        icon: '🇨🇭',
        providerType: 'api',
        baseUrl: 'https://www.zefix.admin.ch/ZefixREST/api/v1',
        description: 'Schweizer Handelsregister',
        requiresApiKey: true,
        defaultConfig: {
            endpoints: {
                search: '/firm/search',
                detail: '/firm/detail'
            }
        }
    },
    
    // GraphQL-Dienste
    'github-graphql': {
        name: 'GitHub GraphQL',
        icon: '🐙',
        providerType: 'graphql',
        baseUrl: 'https://api.github.com/graphql',
        description: 'GitHub GraphQL API',
        requiresApiKey: true,
        defaultConfig: {
            queries: {
                searchRepositories: `
                    query SearchRepositories($query: String!, $limit: Int!) {
                        search(query: $query, type: REPOSITORY, first: $limit) {
                            nodes {
                                ... on Repository {
                                    name
                                    description
                                    url
                                    stargazerCount
                                }
                            }
                        }
                    }
                `
            }
        }
    },
    
    // Datenbank-Dienste
    'postgres': {
        name: 'PostgreSQL',
        icon: '🐘',
        providerType: 'database',
        description: 'PostgreSQL-Datenbank',
        requiresApiKey: false,
        defaultConfig: {
            database_type: 'postgresql',
            connection_string: 'postgresql://user:password@localhost:5432/dbname'
        }
    },
    
    // Web-Scraping-Dienste
    'wikipedia': {
        name: 'Wikipedia',
        icon: '📚',
        providerType: 'web',
        baseUrl: 'https://de.wikipedia.org',
        description: 'Online-Enzyklopädie',
        requiresApiKey: false,
        defaultConfig: {
            selectors: {
                searchResults: '.mw-search-results .mw-search-result',
                title: '.mw-search-result-heading a',
                snippet: '.searchresult'
            }
        }
    },
    
    // Streaming-Dienste
    'twitter': {
        name: 'Twitter API',
        icon: '🐦',
        providerType: 'streaming',
        baseUrl: 'https://api.twitter.com/2',
        description: 'Twitter/X API',
        requiresApiKey: true,
        defaultConfig: {
            streaming_endpoint: '/tweets/search/stream',
            rules_endpoint: '/tweets/search/stream/rules'
        }
    },
    
    // Enterprise-Dienste
    'ldap': {
        name: 'LDAP Directory',
        icon: '🔍',
        providerType: 'enterprise',
        description: 'LDAP-Verzeichnisdienst',
        requiresApiKey: true,
        defaultConfig: {
            enterprise_type: 'ldap',
            ldap_server: 'ldap://ldap.example.com:389',
            base_dn: 'dc=example,dc=com'
        }
    }
};

/**
 * Gibt eine Liste aller verfügbaren Provider-Vorlagen zurück
 */
export function getProviderTemplatesList() {
    return Object.entries(PROVIDER_TEMPLATES).map(([id, template]) => ({
        id,
        name: template.name,
        icon: template.icon,
        description: template.description,
        requiresApiKey: template.requiresApiKey
    }));
}

/**
 * Gibt eine Provider-Vorlage anhand der ID zurück
 */
export function getProviderTemplate(id) {
    return PROVIDER_TEMPLATES[id];
} 