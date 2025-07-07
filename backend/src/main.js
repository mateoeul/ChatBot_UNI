import { tool, agent } from "llamaindex";
import { Ollama } from "@llamaindex/ollama";
import { z } from "zod";
import { empezarChat } from "./lib/cli-chat.js";
import pdfParse from "pdf-parse/lib/pdf-parse.js"
import fs from "fs";
import path from "path";
//import pdfParse from "pdf-parse";


// Configuración
const DEBUG = true;

// System prompt básico
const systemPrompt = `
Sos un asistente de orientación vocacional que brinda una posible carrera universitaraia en base a gustos e intereses del usuario. 
Haces preguntas personales para descubrir sus gustos. 
Además, respondes acerca de información de carreras y universidades de CABA, y extraes esta infromación del pdf.


Respondé de forma clara y breve.
`.trim();


const ollamaLLM = new Ollama({
    model: "qwen3:1.7b",
    temperature: 0.75,
    timeout: 2 * 60 * 1000, // Timeout de 2 minutos
});




// Mensaje de bienvenida
const mensajeBienvenida = `
¡Hola! Soy tu asistente vocacional.
¿Qué necesitás?
`;


// Iniciar el chat


// --- RAG: Extracción e indexado del PDF ---
let universidades = {};
let carreras = {};
let pdfTexto = "";

async function procesarPDF() {
    const dataBuffer = fs.readFileSync(pdfPath);
    const data = await pdfParse(dataBuffer);
    pdfTexto = data.text;
    // Indexar por Universidad y Carrera (asumiendo formato: Universidad:..., Carrera:...)
    const lineas = pdfTexto.split("\n");
    for (const linea of lineas) {
        const uniMatch = linea.match(/Universidad\s*:\s*(.+)/i);
        const carreraMatch = linea.match(/Carrera\s*:\s*(.+)/i);
        if (uniMatch) {
            const uni = uniMatch[1].trim();
            if (!universidades[uni]) universidades[uni] = [];
        }
        if (carreraMatch) {
            const carrera = carreraMatch[1].trim();
            if (!carreras[carrera]) carreras[carrera] = [];
        }
        // Relacionar universidad y carrera si aparecen en la misma línea
        if (uniMatch && carreraMatch) {
            const uni = uniMatch[1].trim();
            const carrera = carreraMatch[1].trim();
            universidades[uni].push(carrera);
            carreras[carrera].push(uni);
        }
    }
}

// --- Tools RAG ---
const buscarPorUniversidadTool = tool({
    name: "buscarPorUniversidad",
    description: "Busca carreras asociadas a una universidad en el PDF de universidades.",
    inputSchema: z.object({ universidad: z.string() }),
    async func({ universidad }) {
        if (!pdfTexto) await procesarPDF();
        const carrerasUni = universidades[universidad];
        if (!carrerasUni || carrerasUni.length === 0) {
            return `No se encontraron carreras para la universidad: ${universidad}`;
        }
        return `Carreras en ${universidad}:\n- ${carrerasUni.join("\n- ")}`;
    }
});

const buscarPorCarreraTool = tool({
    name: "buscarPorCarrera",
    description: "Busca universidades que ofrecen una carrera específica en el PDF de universidades.",
    inputSchema: z.object({ carrera: z.string() }),
    async func({ carrera }) {
        if (!pdfTexto) await procesarPDF();
        const unisCarrera = carreras[carrera];
        if (!unisCarrera || unisCarrera.length === 0) {
            return `No se encontraron universidades para la carrera: ${carrera}`;
        }
        return `Universidades que ofrecen ${carrera}:\n- ${unisCarrera.join("\n- ")}`;
    }
});

const listarUniversidadesTool = tool({
    name: "listarUniversidades",
    description: "Lista todas las universidades encontradas en el PDF.",
    inputSchema: z.object({}),
    async func() {
        if (!pdfTexto) await procesarPDF();
        return Object.keys(universidades).length > 0
            ? `Universidades:\n- ${Object.keys(universidades).join("\n- ")}`
            : "No se encontraron universidades.";
    }
});

const listarCarrerasTool = tool({
    name: "listarCarreras",
    description: "Lista todas las carreras encontradas en el PDF.",
    inputSchema: z.object({}),
    async func() {
        if (!pdfTexto) await procesarPDF();
        return Object.keys(carreras).length > 0
            ? `Carreras:\n- ${Object.keys(carreras).join("\n- ")}`
            : "No se encontraron carreras.";
    }
});

// --- Agente con tools RAG ---
const elAgente = agent({
    tools: [buscarPorUniversidadTool, buscarPorCarreraTool, listarUniversidadesTool, listarCarrerasTool],
    llm: ollamaLLM,
    systemPrompt: systemPrompt,
});

empezarChat(elAgente, mensajeBienvenida);

export { elAgente };