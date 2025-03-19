/**
 * Files API Module
 * 
 * Provides functions for interacting with the files API
 */
const FilesAPI = {
    /**
     * Get all files uploaded by the user
     * @returns {Promise} Promise that resolves to the list of files
     */
    getFiles: async function() {
        try {
            const response = await fetch('/api/models/files/');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching files:', error);
            throw error;
        }
    },
    
    /**
     * Upload a file
     * @param {File} file - The file to upload
     * @returns {Promise} Promise that resolves to the uploaded file data
     */
    uploadFile: async function(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/models/files/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error uploading file:', error);
            throw error;
        }
    },
    
    /**
     * Delete a file
     * @param {string} fileId - The ID of the file to delete
     * @returns {Promise} Promise that resolves when the file is deleted
     */
    deleteFile: async function(fileId) {
        try {
            const response = await fetch(`/api/models/files/${fileId}/`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return true;
        } catch (error) {
            console.error('Error deleting file:', error);
            throw error;
        }
    },
    
    /**
     * Process a file with AI
     * @param {string} fileId - The ID of the file to process
     * @returns {Promise} Promise that resolves to the processing results
     */
    processFile: async function(fileId) {
        try {
            const response = await fetch(`/files/process/${fileId}/`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error processing file:', error);
            throw error;
        }
    }
};

// Export the module
export default FilesAPI; 