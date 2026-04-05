const express = require('express');
const cors = require('cors');
const { execFile } = require('child_process');
const { Groq } = require('groq-sdk');
require('dotenv').config();

const app = express();

// Explicitly allow all origins including file:// protocol from Unity WebGL
app.use(cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());

const PORT = process.env.PORT || 3000;
const PYTHON_QUERY_TIMEOUT_MS = Number(process.env.PYTHON_QUERY_TIMEOUT_MS || 120000);
const FACULTY_COLLECTION = process.env.FACULTY_COLLECTION_NAME || 'faculty_collection';
const PLACEMENTS_COLLECTION = process.env.PLACEMENTS_COLLECTION_NAME || 'placements_collection';
const COLLEGE_INFO_COLLECTION = process.env.COLLEGE_INFO_COLLECTION_NAME || 'college_info_collection';

// Initialize Groq client
const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
});

function detectCollection(query) {
    const q = query.toLowerCase();

    const facultyKeywords = [
        'faculty', 'professor', 'assistant professor', 'associate professor', 'hod',
        'dean', 'department', 'teacher', 'lecturer', 'qualification', 'staff'
    ];
    const placementKeywords = [
        'placement', 'placements', 'package', 'ctc', 'lpa', 'recruiter', 'recruiters',
        'offer', 'offers', 'stipend', 'company', 'companies', 'internship', 'batch'
    ];
    const collegeInfoKeywords = [
        'college', 'institute', 'history', 'location', 'address', 'campus', 'rating',
        'ranking', 'nirf', 'naac', 'nba', 'accreditation', 'hostel', 'library', 'sports',
        'facility', 'facilities', 'club', 'clubs', 'contact', 'website', 'email', 'phone'
    ];

    if (placementKeywords.some(keyword => q.includes(keyword))) {
        return { name: PLACEMENTS_COLLECTION, topic: 'placements' };
    }

    if (facultyKeywords.some(keyword => q.includes(keyword))) {
        return { name: FACULTY_COLLECTION, topic: 'faculty' };
    }

    if (collegeInfoKeywords.some(keyword => q.includes(keyword))) {
        return { name: COLLEGE_INFO_COLLECTION, topic: 'college_info' };
    }

    return { name: COLLEGE_INFO_COLLECTION, topic: 'college_info' };
}

app.post('/query', async (req, res) => {
    try {
        const { query } = req.body;
        if (!query || typeof query !== 'string' || !query.trim()) {
            return res.status(400).json({ error: "Query is required" });
        }

        const sanitizedQuery = query.trim();
        const targetCollection = detectCollection(sanitizedQuery);
        console.log(`[Query] ${sanitizedQuery}`);
        console.log(`[Routing] ${targetCollection.topic} -> ${targetCollection.name}`);

        execFile(
            'python',
            ['chroma_client.py', sanitizedQuery, targetCollection.name],
            { timeout: PYTHON_QUERY_TIMEOUT_MS, cwd: __dirname },
            async (error, stdout, stderr) => {
                let searchResult;
                try {
                    const lines = stdout.trim().split('\n').filter(l => l.trim());
                    const jsonLine = lines[lines.length - 1];
                    searchResult = JSON.parse(jsonLine);
                } catch (parseErr) {
                    if (error) {
                        console.error(`[Python Error] ${error.message}`);
                    }
                    if (stderr) {
                        console.error(`[Python Stderr] ${stderr}`);
                    }
                    console.error("[Parse Error] Python output:", stdout);
                    return res.status(500).json({ error: "Invalid vector search response" });
                }

                if (error) {
                    console.error(`[Python Error] ${error.message}`);
                }
                if (stderr && stderr.trim()) {
                    console.warn(`[Python Stderr] ${stderr}`);
                }

                if (searchResult.status !== "success") {
                    console.error("[Search Error]", searchResult.message);
                    return res.status(500).json({ error: searchResult.message || "Vector search failed" });
                }

                const retrievedData = Array.isArray(searchResult.data) ? searchResult.data : [];
                console.log(`[ChromaDB] Retrieved ${retrievedData.length} results`);

                // Build context string from retrieved documents
                const contextStr = retrievedData.length > 0
                    ? retrievedData.map((item, i) => `[${i + 1}] ${item.document}`).join('\n')
                    : "No relevant faculty information found.";

                // Construct RAG prompt
                const prompt = `You are a smart college assistant chatbot for a virtual campus tour.
Answer ONLY using the context provided below. If the answer is not in the context, say "I don't have that information."
The current query topic is: ${targetCollection.topic}.

Context:
${contextStr}

Question: ${sanitizedQuery}

Give a clear, helpful, and friendly answer.`;

                try {
                    console.log("[Groq] Sending request...");
                    const completion = await groq.chat.completions.create({
                        messages: [
                            {
                                role: "system",
                                content: "You are a helpful and knowledgeable college assistant chatbot for a virtual campus tour. Answer only from the given context."
                            },
                            { role: "user", content: prompt }
                        ],
                        model: "llama-3.1-8b-instant",
                        temperature: 0.1,
                        max_tokens: 512
                    });

                    const answer = completion.choices[0].message.content;
                    console.log("[Groq] Response received.");

                    return res.json({
                        response: answer,
                        topic: targetCollection.topic,
                        sources: retrievedData.map(d => d.metadata)
                    });
                } catch (groqErr) {
                    console.error("[Groq Error]", groqErr.message);
                    return res.status(500).json({ error: "LLM generation failed", details: groqErr.message });
                }
            }
        );

    } catch (err) {
        console.error("[Server Error]", err.message);
        return res.status(500).json({ error: "Internal server error" });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        model: 'llama-3.1-8b-instant',
        database: process.env.CHROMA_CLOUD_DATABASE || null,
        collections: {
            faculty: FACULTY_COLLECTION,
            placements: PLACEMENTS_COLLECTION,
            college_info: COLLEGE_INFO_COLLECTION
        }
    });
});

app.listen(PORT, () => {
    console.log(`✅ Server running at http://localhost:${PORT}`);
    console.log(`   POST /query — RAG chatbot endpoint`);
    console.log(`   GET  /health — Health check`);
});
