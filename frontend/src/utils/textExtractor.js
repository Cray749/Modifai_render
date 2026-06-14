/**
 * Client-side text extraction for PDF, DOCX, and TXT files.
 * Extracts text from RANDOM positions within files to give
 * a representative sample for data quality evaluation.
 */
import * as pdfjsLib from 'pdfjs-dist'
import mammoth from 'mammoth'

// Point pdf.js to the bundled worker using Vite's native Web Worker support (?worker).
// This forces Vite to output a standard .js chunk instead of .mjs, preventing
// deployment servers from rejecting the script due to 'application/octet-stream' MIME issues.
import PdfWorker from 'pdfjs-dist/build/pdf.worker.mjs?worker'
pdfjsLib.GlobalWorkerOptions.workerPort = new PdfWorker()

const SAMPLE_CHARS = 2000

/**
 * Pick up to `count` random files from the array, preferring text-extractable types.
 * Skips image files (PNG/JPG) since we can't extract text from them client-side.
 */
export function pickRandomFiles(fileObjects, count = 3) {
    // Filter to text-extractable types only
    const extractable = fileObjects.filter((f) => {
        const ext = f.name.split('.').pop()?.toLowerCase()
        return ['txt', 'pdf', 'docx'].includes(ext)
    })

    if (extractable.length === 0) {
        throw new Error(
            'None of your uploaded files can be evaluated for text quality. ' +
            'Data quality evaluation supports .txt, .pdf, and .docx files. ' +
            'Image files (PNG/JPG) cannot be evaluated — you can skip this step.'
        )
    }

    // Shuffle and take up to `count`
    const shuffled = [...extractable].sort(() => Math.random() - 0.5)
    return shuffled.slice(0, Math.min(count, shuffled.length))
}

/**
 * Extract ~SAMPLE_CHARS of text from a RANDOM position within a file.
 * This avoids always sampling the intro/header and gives a more
 * representative view of overall content quality.
 */
export async function extractTextSample(file) {
    const ext = file.name.split('.').pop()?.toLowerCase()

    switch (ext) {
        case 'txt':
            return extractFromTxt(file)
        case 'pdf':
            return extractFromPdf(file)
        case 'docx':
            return extractFromDocx(file)
        default:
            throw new Error(
                `Cannot extract text from .${ext} files. ` +
                `Evaluation supports .txt, .pdf, and .docx only.`
            )
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Pick a random slice of `size` chars from `fullText`. */
function randomSlice(fullText, size = SAMPLE_CHARS) {
    if (fullText.length <= size) return fullText

    // Pick a random start position, avoiding the very end
    const maxStart = fullText.length - size
    const start = Math.floor(Math.random() * maxStart)
    return fullText.substring(start, start + size)
}

// ── TXT ──────────────────────────────────────────────────────────────────────
async function extractFromTxt(file) {
    const fullText = await file.text()
    return randomSlice(fullText)
}

// ── PDF ──────────────────────────────────────────────────────────────────────
async function extractFromPdf(file) {
    const arrayBuffer = await file.arrayBuffer()
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise
    const totalPages = pdf.numPages

    if (totalPages === 0) {
        throw new Error('PDF has no pages.')
    }

    // Pick a random starting page (not always page 1)
    const startPage = Math.floor(Math.random() * totalPages) + 1
    const pagesToRead = Math.min(5, totalPages) // read up to 5 pages from the random start

    let extracted = ''
    for (let i = 0; i < pagesToRead; i++) {
        // Wrap around if we go past the last page
        const pageNum = ((startPage - 1 + i) % totalPages) + 1
        const page = await pdf.getPage(pageNum)
        const content = await page.getTextContent()
        const pageText = content.items.map((item) => item.str).join(' ')
        extracted += pageText + '\n'

        if (extracted.length >= SAMPLE_CHARS) break
    }

    const result = extracted.substring(0, SAMPLE_CHARS).trim()
    if (!result) {
        throw new Error(
            'Could not extract text from this PDF. ' +
            'It may be a scanned/image-only PDF. Try a text-based PDF or .txt file.'
        )
    }
    return result
}

// ── DOCX ─────────────────────────────────────────────────────────────────────
async function extractFromDocx(file) {
    const arrayBuffer = await file.arrayBuffer()
    const result = await mammoth.extractRawText({ arrayBuffer })
    const fullText = result.value.trim()

    if (!fullText) {
        throw new Error(
            'Could not extract text from this DOCX. The document may be empty or image-only.'
        )
    }
    return randomSlice(fullText)
}
