import { html, css, LitElement } from 'lit';
import { repeat } from 'lit/directives/repeat.js';

export class SpreadsheetViewer extends LitElement {
    static properties = {
        data: { type: Array },
        activeSheet: { type: String },
        filters: { type: Object },
        sortColumn: { type: String },
        sortDirection: { type: String }
    };

    constructor() {
        super();
        this.data = [];
        this.activeSheet = '';
        this.filters = {};
        this.sortColumn = '';
        this.sortDirection = 'asc';
    }

    sortData(data) {
        if (!this.sortColumn) return data;
        
        return [...data].sort((a, b) => {
            const aVal = a[this.sortColumn];
            const bVal = b[this.sortColumn];
            const direction = this.sortDirection === 'asc' ? 1 : -1;
            
            return aVal > bVal ? direction : -direction;
        });
    }

    filterData(data) {
        return data.filter(row => {
            return Object.entries(this.filters).every(([key, value]) => {
                if (!value) return true;
                return String(row[key]).includes(String(value));
            });
        });
    }

    render() {
        const headers = this.data[0] ? Object.keys(this.data[0]) : [];
        const processedData = this.filterData(this.sortData(this.data));

        return html`
            <div class="spreadsheet-viewer">
                <div class="toolbar">
                    <div class="filters">
                        ${headers.map(header => html`
                            <input 
                                placeholder="Filter ${header}"
                                .value="${this.filters[header] || ''}"
                                @input="${e => this.filters = {
                                    ...this.filters,
                                    [header]: e.target.value
                                }}"
                            >
                        `)}
                    </div>
                </div>

                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                ${headers.map(header => html`
                                    <th @click="${() => {
                                        this.sortColumn = header;
                                        this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
                                    }}">
                                        ${header}
                                        ${this.sortColumn === header ? 
                                            (this.sortDirection === 'asc' ? '↑' : '↓') : ''}
                                    </th>
                                `)}
                            </tr>
                        </thead>
                        <tbody>
                            ${repeat(processedData, (row, i) => i, row => html`
                                <tr>
                                    ${headers.map(header => html`
                                        <td>${row[header]}</td>
                                    `)}
                                </tr>
                            `)}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    static styles = css`
        :host {
            display: block;
        }

        .spreadsheet-viewer {
            height: 100%;
            display: flex;
            flex-direction: column;
        }

        .toolbar {
            padding: 1rem;
            background: var(--toolbar-bg, #f5f5f5);
        }

        .filters {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .table-container {
            overflow: auto;
            flex: 1;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 0.5rem;
            border: 1px solid var(--border-color, #ddd);
        }

        th {
            background: var(--header-bg, #f0f0f0);
            cursor: pointer;
        }

        tr:nth-child(even) {
            background: var(--row-alt-bg, #f9f9f9);
        }
    `;
}

customElements.define('spreadsheet-viewer', SpreadsheetViewer); 