import ky from 'ky'

// ─── Base API Client ──────────────────────────────────────────────────────────
// All HTTP requests go through this configured ky instance.
// Update prefixUrl when the backend URL changes (e.g. staging, production).

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1'

export const api = ky.create({
    prefixUrl: API_BASE,
    timeout: 30_000,
    retry: {
        limit: 2,
        methods: ['get'],          // only retry idempotent requests
        statusCodes: [408, 502, 503, 504],
    },
})


// ─── SSE (Server-Sent Events) ─────────────────────────────────────────────────
// Use for real-time pipeline progress streaming.
//
// Usage:
//   const unsubscribe = subscribeToStream('/projects/proj-001/stream', {
//       onMessage: (data) => console.log(data),
//       onError:   (err)  => console.error(err),
//   })
//   // Later: unsubscribe()

export function subscribeToStream(path, { onMessage, onError, onOpen } = {}) {
    const url = `${API_BASE}/${path.replace(/^\//, '')}`
    const source = new EventSource(url)

    if (onOpen) {
        source.addEventListener('open', onOpen)
    }

    source.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data)
            onMessage?.(data)
        } catch {
            // Non-JSON message, pass raw data
            onMessage?.(event.data)
        }
    }

    source.onerror = (event) => {
        onError?.(event)
        // EventSource auto-reconnects on error; close if needed
    }

    // Return cleanup function
    return () => {
        source.close()
    }
}


// ─── Convenience Helpers ──────────────────────────────────────────────────────
// Typed wrappers around the ky instance for common patterns.

export const apiClient = {
    get: (path, options) => api.get(path, options).json(),
    post: (path, body, options) => api.post(path, { json: body, ...options }).json(),
    put: (path, body, options) => api.put(path, { json: body, ...options }).json(),
    patch: (path, body, options) => api.patch(path, { json: body, ...options }).json(),
    delete: (path, options) => api.delete(path, options).json(),
    upload: (path, formData, options) => api.post(path, { body: formData, ...options }).json(),
}

export default apiClient
