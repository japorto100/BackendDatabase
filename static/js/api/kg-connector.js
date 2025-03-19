/**
 * Knowledge Graph Connector API
 * 
 * Provides interface to KG functionality for the frontend
 */

/**
 * Get a KG connector instance for the current session
 * @returns {Promise<Object>} KG connector object with methods
 */
export async function getKGConnector() {
    try {
        const response = await fetch('/api/kg-connector/');
        if (!response.ok) {
            console.error('Error getting KG connector:', response.statusText);
            return null;
        }
        
        const data = await response.json();
        
        // Return object with KG methods
        return {
            /**
             * Enhance a search query with KG knowledge
             * @param {string} query - Original search query
             * @returns {Promise<string>} Enhanced query
             */
            enhance_search_query: async (query) => {
                try {
                    const response = await fetch('/api/kg/enhance-query/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': getCsrfToken()
                        },
                        body: JSON.stringify({ query })
                    });
                    
                    if (!response.ok) {
                        console.warn('Error enhancing query with KG:', response.statusText);
                        return query; // Return original query as fallback
                    }
                    
                    const data = await response.json();
                    return data.enhanced_query;
                } catch (error) {
                    console.error('Error in enhance_search_query:', error);
                    return query; // Return original query as fallback
                }
            },
            
            /**
             * Store valuable search results in the knowledge graph
             * @param {string} query - Original query
             * @param {Array} results - Search results to store
             * @returns {Promise<boolean>} Success status
             */
            store_search_results: async (query, results) => {
                try {
                    // Only store high-quality results
                    const valuableResults = results.filter(r => 
                        r.score > 0.7 || r.relevance > 0.7
                    );
                    
                    if (valuableResults.length === 0) {
                        return false;
                    }
                    
                    const response = await fetch('/api/kg/store-search-results/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': getCsrfToken()
                        },
                        body: JSON.stringify({ 
                            query,
                            results: valuableResults.slice(0, 5) // Limit to top 5
                        })
                    });
                    
                    return response.ok;
                } catch (error) {
                    console.error('Error storing search results in KG:', error);
                    return false;
                }
            },
            
            /**
             * Get entities from the knowledge graph related to a query
             * @param {string} query - The query to find related entities for
             * @returns {Promise<Array>} Related entities
             */
            get_related_entities: async (query) => {
                try {
                    const response = await fetch(`/api/kg/entities/?query=${encodeURIComponent(query)}`);
                    
                    if (!response.ok) {
                        return [];
                    }
                    
                    const data = await response.json();
                    return data.entities || [];
                } catch (error) {
                    console.error('Error getting related entities:', error);
                    return [];
                }
            }
        };
    } catch (error) {
        console.error('Error initializing KG connector:', error);
        return null;
    }
}

/**
 * Get CSRF token from cookies
 * @returns {string} CSRF token
 */
function getCsrfToken() {
    const name = 'csrftoken=';
    const decodedCookie = decodeURIComponent(document.cookie);
    const cookieArray = decodedCookie.split(';');
    
    for (let i = 0; i < cookieArray.length; i++) {
        let cookie = cookieArray[i].trim();
        if (cookie.indexOf(name) === 0) {
            return cookie.substring(name.length, cookie.length);
        }
    }
    
    return '';
}

export default { getKGConnector }; 