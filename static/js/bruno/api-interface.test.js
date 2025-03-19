import { APIInterface } from './api-interface.js';

describe('APIInterface', () => {
    let apiInterface;

    beforeEach(() => {
        apiInterface = new APIInterface();
    });

    test('detectBottlenecks identifies slow queries', () => {
        apiInterface.responseData = {
            debug: {
                sql_queries: [
                    { duration: 150, sql: 'SELECT * FROM table' }
                ]
            }
        };
        const bottlenecks = apiInterface.detectBottlenecks();
        expect(bottlenecks).toContainEqual(
            expect.objectContaining({
                type: 'sql',
                severity: 'warning'
            })
        );
    });

    // ... weitere Tests ...
}); 