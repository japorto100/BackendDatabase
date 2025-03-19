import { html, css, LitElement } from 'lit';
import { repeat } from 'lit/directives/repeat.js';

class ProviderManagement extends LitElement {
    static properties = {
        providers: { type: Array },
        loading: { type: Boolean },
        error: { type: String }
    };

    constructor() {
        super();
        this.providers = [];
        this.loading = true;
        this.error = null;
    }

    connectedCallback() {
        super.connectedCallback();
        this.loadProviders();
    }

    async loadProviders() {
        try {
            const response = await fetch('/api/providers/');
            if (!response.ok) {
                throw new Error('Failed to load providers');
            }
            
            this.providers = await response.json();
        } catch (error) {
            this.error = error.message;
        } finally {
            this.loading = false;
        }
    }

    async updateProvider(id, data) {
        try {
            const response = await fetch(`/api/providers/${id}/`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this._getCsrfToken()
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Failed to update provider');
            }
            
            // Aktualisiere den Provider in der Liste
            const updatedProvider = await response.json();
            this.providers = this.providers.map(p => 
                p.id === updatedProvider.id ? updatedProvider : p
            );
            
            return true;
        } catch (error) {
            console.error('Error updating provider:', error);
            return false;
        }
    }

    async deleteProvider(id) {
        if (!confirm('Möchten Sie diesen Provider wirklich löschen?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/providers/${id}/`, {
                method: 'DELETE',
                headers: {
                    'X-CSRFToken': this._getCsrfToken()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete provider');
            }
            
            // Entferne den Provider aus der Liste
            this.providers = this.providers.filter(p => p.id !== id);
            
            return true;
        } catch (error) {
            console.error('Error deleting provider:', error);
            return false;
        }
    }

    _getCsrfToken() {
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

    _editProvider(provider) {
        const dialog = document.createElement('dialog');
        dialog.innerHTML = `
            <form id="edit-provider-form">
                <h2>Provider bearbeiten</h2>
                
                <div class="form-group">
                    <label for="provider-name">Name</label>
                    <input type="text" id="provider-name" value="${provider.name}" required>
                </div>
                
                <div class="form-group">
                    <label for="provider-api-key">API Key</label>
                    <input type="password" id="provider-api-key" value="${provider.api_key || ''}">
                    <button type="button" id="validate-key-btn">API-Key validieren</button>
                    <div id="validation-result"></div>
                </div>
                
                <div class="form-group">
                    <label for="provider-base-url">Base URL</label>
                    <input type="url" id="provider-base-url" value="${provider.base_url || ''}">
                </div>
                
                <div class="form-group">
                    <label for="provider-headers">Custom Headers (JSON)</label>
                    <textarea id="provider-headers" rows="3">${
                        provider.custom_headers ? JSON.stringify(provider.custom_headers, null, 2) : ''
                    }</textarea>
                </div>
                
                <div class="form-group">
                    <label for="provider-config">Config (JSON)</label>
                    <textarea id="provider-config" rows="5">${
                        provider.config ? JSON.stringify(provider.config, null, 2) : ''
                    }</textarea>
                </div>
                
                <div class="form-group">
                    <label for="provider-active">
                        <input type="checkbox" id="provider-active" ${provider.is_active ? 'checked' : ''}>
                        Aktiv
                    </label>
                </div>
                
                <div class="form-actions">
                    <button type="button" id="cancel-btn">Abbrechen</button>
                    <button type="submit" id="save-btn">Speichern</button>
                </div>
            </form>
        `;
        
        document.body.appendChild(dialog);
        dialog.showModal();
        
        // Event Listeners
        dialog.querySelector('#cancel-btn').addEventListener('click', () => {
            dialog.close();
            dialog.remove();
        });
        
        // API-Key Validierung
        dialog.querySelector('#validate-key-btn').addEventListener('click', async () => {
            const apiKey = dialog.querySelector('#provider-api-key').value;
            const baseUrl = dialog.querySelector('#provider-base-url').value;
            const validationResult = dialog.querySelector('#validation-result');
            
            if (apiKey) {
                const validateBtn = dialog.querySelector('#validate-key-btn');
                validateBtn.disabled = true;
                validateBtn.textContent = 'Validiere...';
                
                try {
                    const response = await fetch('/api/validate-api-key/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': this._getCsrfToken()
                        },
                        body: JSON.stringify({
                            provider_type: provider.provider_type,
                            base_url: baseUrl,
                            api_key: apiKey
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        validationResult.className = 'validation-success';
                        validationResult.textContent = data.message;
                    } else {
                        validationResult.className = 'validation-error';
                        validationResult.textContent = data.error;
                    }
                } catch (error) {
                    validationResult.className = 'validation-error';
                    validationResult.textContent = 'Fehler bei der Validierung: ' + error.message;
                } finally {
                    validateBtn.disabled = false;
                    validateBtn.textContent = 'API-Key validieren';
                }
            }
        });
        
        // Formular-Submission
        dialog.querySelector('#edit-provider-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            let customHeaders = {};
            let config = {};
            
            try {
                const headersText = dialog.querySelector('#provider-headers').value;
                if (headersText) {
                    customHeaders = JSON.parse(headersText);
                }
                
                const configText = dialog.querySelector('#provider-config').value;
                if (configText) {
                    config = JSON.parse(configText);
                }
            } catch (error) {
                alert('Ungültiges JSON-Format in Headers oder Config');
                return;
            }
            
            const updatedProvider = {
                name: dialog.querySelector('#provider-name').value,
                api_key: dialog.querySelector('#provider-api-key').value,
                base_url: dialog.querySelector('#provider-base-url').value,
                custom_headers: customHeaders,
                config: config,
                is_active: dialog.querySelector('#provider-active').checked
            };
            
            const success = await this.updateProvider(provider.id, updatedProvider);
            
            if (success) {
                dialog.close();
                dialog.remove();
            } else {
                alert('Fehler beim Aktualisieren des Providers');
            }
        });
    }

    render() {
        if (this.loading) {
            return html`<div class="loading">Lade Provider...</div>`;
        }
        
        if (this.error) {
            return html`<div class="error">Fehler: ${this.error}</div>`;
        }
        
        return html`
            <div class="provider-management">
                <h2>Provider-Verwaltung</h2>
                
                <table class="provider-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Typ</th>
                            <th>Status</th>
                            <th>API-Key</th>
                            <th>Aktionen</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${repeat(this.providers, (provider) => provider.id, (provider) => html`
                            <tr class="${provider.is_active ? '' : 'inactive'}">
                                <td>${provider.name}</td>
                                <td>${provider.provider_type}</td>
                                <td>
                                    <span class="status ${provider.is_active ? 'active' : 'inactive'}">
                                        ${provider.is_active ? 'Aktiv' : 'Inaktiv'}
                                    </span>
                                    ${!provider.api_key && provider.provider_type !== 'web' ? 
                                        html`<span class="warning">API-Key fehlt</span>` : 
                                        ''
                                    }
                                </td>
                                <td>
                                    ${provider.api_key ? 
                                        html`<span class="key-status">✓ Vorhanden</span>` : 
                                        html`<span class="key-status empty">✗ Fehlt</span>`
                                    }
                                </td>
                                <td>
                                    <button @click=${() => this._editProvider(provider)}>
                                        Bearbeiten
                                    </button>
                                    <button @click=${() => this.deleteProvider(provider.id)} class="delete">
                                        Löschen
                                    </button>
                                </td>
                            </tr>
                        `)}
                    </tbody>
                </table>
            </div>
        `;
    }

    static styles = css`
        .provider-management {
            padding: 1rem;
        }
        
        .provider-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .provider-table th, .provider-table td {
            padding: 0.5rem;
            border: 1px solid #ddd;
            text-align: left;
        }
        
        .provider-table th {
            background-color: #f5f5f5;
        }
        
        .provider-table tr.inactive {
            background-color: #f9f9f9;
            color: #777;
        }
        
        .status {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        
        .status.active {
            background-color: #dff0d8;
            color: #3c763d;
        }
        
        .status.inactive {
            background-color: #f2dede;
            color: #a94442;
        }
        
        .warning {
            display: inline-block;
            margin-left: 0.5rem;
            padding: 0.25rem 0.5rem;
            background-color: #fcf8e3;
            color: #8a6d3b;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        
        .key-status {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            background-color: #dff0d8;
            color: #3c763d;
        }
        
        .key-status.empty {
            background-color: #f2dede;
            color: #a94442;
        }
        
        button {
            padding: 0.25rem 0.5rem;
            margin-right: 0.25rem;
            border: 1px solid #ddd;
            background-color: #f5f5f5;
            cursor: pointer;
        }
        
        button.delete {
            background-color: #f2dede;
            color: #a94442;
        }
        
        .validation-success {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background-color: #dff0d8;
            color: #3c763d;
            border-radius: 4px;
        }
        
        .validation-error {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background-color: #f2dede;
            color: #a94442;
            border-radius: 4px;
        }
    `;
}

customElements.define('perplexica-provider-management', ProviderManagement);

export default ProviderManagement; 