import { tool, agent } from "llamaindex";
import { Ollama } from "@llamaindex/ollama";
import { z } from "zod";
import { empezarChat } from "./lib/cli-chat.js";
import pdfParse from "pdf-parse/lib/pdf-parse.js"
import fs from "fs";
import path from "path";
//import pdfParse from "pdf-parse";
//ciro

// Corregir la obtención de la ruta absoluta del PDF para ES modules
const __filename = new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
const __dirname = path.dirname(__filename);
const pdfPath = path.resolve(__dirname, '../data/PDF-UNIVERSE.pdf');
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
let universidadesSet = new Set();
let carreras = {};
let pdfTexto = "";

try {
    console.log('Intentando leer:', pdfPath);
    const testBuffer = fs.readFileSync(pdfPath);
    console.log('Lectura exitosa, tamaño:', testBuffer.length);
  } catch (e) {
    console.error('Error leyendo el PDF:', e);
  }


async function procesarPDF() {
    const dataBuffer = fs.readFileSync(pdfPath);
    const data = await pdfParse(dataBuffer);
    pdfTexto = data.text;
    // Log para depuración
    //console.log('Primeros 500 caracteres del PDF extraído:', pdfTexto.slice(0, 500));
    // Indexar universidades: líneas que contienen 'Universidad'
    const lineas = pdfTexto.split("\n");
    for (const linea of lineas) {
        if (linea.match(/universidad/i)) {
            const nombre = linea.trim();
            if (nombre.length > 5) universidadesSet.add(nombre);
        }
    }
}

// --- Tools RAG ---
const buscarPorUniversidadTool = tool({
    name: "buscarPorUniversidad",
    description: "Busca carreras asociadas a una universidad en el PDF de universidades.",
    parameters: z.object({ universidad: z.string() }),
    execute: async ({ universidad }) => {
        if (!pdfTexto) await procesarPDF();
        const carrerasUni = universidadesSet.has(universidad) ? [universidad] : [];
        if (!carrerasUni || carrerasUni.length === 0) {
            return `No se encontraron carreras para la universidad: ${universidad}`;
        }
        return `Carreras en ${universidad}:\n- ${carrerasUni.join("\n- ")}`;
    }
});

const buscarPorCarreraTool = tool({
    name: "buscarPorCarrera",
    description: "Busca universidades que ofrecen una carrera específica en el PDF de universidades.",
    parameters: z.object({ carrera: z.string() }),
    execute: async ({ carrera }) => {
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
    parameters: z.object({}),
    execute: async () => {
        if (!pdfTexto) await procesarPDF();
        const universidades = Array.from(universidadesSet);
        return universidades.length > 0
            ? `Universidades:\n- ${universidades.join("\n- ")}`
            : "No se encontraron universidades.";
    }
});

const listarCarrerasTool = tool({
    name: "listarCarreras",
    description: "Lista todas las carreras encontradas en el PDF.",
    parameters: z.object({}),
    execute: async () => {
        if (!pdfTexto) await procesarPDF();
        return Object.keys(carreras).length > 0
            ? `Carreras:\n- ${Object.keys(carreras).join("\n- ")}`
            : "No se encontraron carreras.";
    }
});

// --- Tools avanzadas ---

// Buscar universidades por palabra clave
const buscarUniversidadPorPalabraTool = tool({
    name: "buscarUniversidadPorPalabra",
    description: "Busca universidades cuyo nombre contenga una palabra clave.",
    parameters: z.object({ palabra: z.string() }),
    execute: async ({ palabra }) => {
        if (!pdfTexto) await procesarPDF();
        const universidades = Array.from(universidadesSet).filter(u => u.toLowerCase().includes(palabra.toLowerCase()));
        return universidades.length > 0
            ? `Universidades que contienen '${palabra}':\n- ${universidades.join("\n- ")}`
            : `No se encontraron universidades que contengan '${palabra}'.`;
    }
});

// Mostrar información detallada de una universidad
const detalleUniversidadTool = tool({
    name: "detalleUniversidad",
    description: "Muestra información detallada de una universidad buscando su nombre en el PDF.",
    parameters: z.object({ universidad: z.string() }),
    execute: async ({ universidad }) => {
        if (!pdfTexto) await procesarPDF();
        // Busca el bloque de texto que contiene el nombre de la universidad
        const regex = new RegExp(`(^|\n)(.*${universidad}.*)(\n[\s\S]*?)(?=\n[A-Z][^\n]+\n|$)`, 'i');
        const match = pdfTexto.match(regex);
        return match ? `Detalle de ${universidad}:\n${match[0].trim()}` : `No se encontró información detallada para ${universidad}.`;
    }
});

// Comparar dos universidades
const compararUniversidadesTool = tool({
    name: "compararUniversidades",
    description: "Compara dos universidades por nombre, mostrando sus descripciones si están disponibles.",
    parameters: z.object({ universidad1: z.string(), universidad2: z.string() }),
    execute: async ({ universidad1, universidad2 }) => {
        if (!pdfTexto) await procesarPDF();
        const detalle1 = await detalleUniversidadTool.execute({ universidad: universidad1 });
        const detalle2 = await detalleUniversidadTool.execute({ universidad: universidad2 });
        return `Comparación:\n\n${detalle1}\n\n---\n\n${detalle2}`;
    }
});

// Buscar carreras por palabra clave (búsqueda textual)
const buscarCarreraPorPalabraTool = tool({
    name: "buscarCarreraPorPalabra",
    description: "Busca carreras cuyo nombre contenga una palabra clave (búsqueda textual en el PDF).",
    parameters: z.object({ palabra: z.string() }),
    execute: async ({ palabra }) => {
        if (!pdfTexto) await procesarPDF();
        // Busca líneas que contengan la palabra
        const lineas = pdfTexto.split("\n");
        const carrerasEncontradas = lineas.filter(l => l.toLowerCase().includes(palabra.toLowerCase()) && l.length > 5);
        return carrerasEncontradas.length > 0
            ? `Carreras que contienen '${palabra}':\n- ${carrerasEncontradas.join("\n- ")}`
            : `No se encontraron carreras que contengan '${palabra}'.`;
    }
});

// Mostrar información detallada de una carrera
const detalleCarreraTool = tool({
    name: "detalleCarrera",
    description: "Muestra información detallada de una carrera buscando su nombre en el PDF.",
    parameters: z.object({ carrera: z.string() }),
    execute: async ({ carrera }) => {
        if (!pdfTexto) await procesarPDF();
        // Busca el bloque de texto que contiene el nombre de la carrera
        const regex = new RegExp(`(^|\n)(.*${carrera}.*)(\n[\s\S]*?)(?=\n[A-Z][^\n]+\n|$)`, 'i');
        const match = pdfTexto.match(regex);
        return match ? `Detalle de ${carrera}:\n${match[0].trim()}` : `No se encontró información detallada para ${carrera}.`;
    }
});

// Comparar dos carreras
const compararCarrerasTool = tool({
    name: "compararCarreras",
    description: "Compara dos carreras mostrando los bloques de texto relacionados si están disponibles.",
    parameters: z.object({ carrera1: z.string(), carrera2: z.string() }),
    execute: async ({ carrera1, carrera2 }) => {
        if (!pdfTexto) await procesarPDF();
        const detalle1 = await detalleCarreraTool.execute({ carrera: carrera1 });
        const detalle2 = await detalleCarreraTool.execute({ carrera: carrera2 });
        return `Comparación de carreras:\n\n${detalle1}\n\n---\n\n${detalle2}`;
    }
});

// Listar carreras de una universidad (búsqueda textual simple)
const listarCarrerasDeUniversidadTool = tool({
    name: "listarCarrerasDeUniversidad",
    description: "Lista carreras mencionadas cerca del nombre de una universidad en el PDF (búsqueda textual simple).",
    parameters: z.object({ universidad: z.string() }),
    execute: async ({ universidad }) => {
        if (!pdfTexto) await procesarPDF();
        // Busca el bloque de texto de la universidad y extrae posibles carreras (líneas siguientes)
        const regex = new RegExp(`(^|\n)(.*${universidad}.*)(\n[\s\S]{0,1000})`, 'i');
        const match = pdfTexto.match(regex);
        if (!match) return `No se encontró información para ${universidad}.`;
        const bloque = match[0];
        // Busca posibles carreras (líneas con palabras clave típicas)
        const carreras = bloque.split("\n").filter(l => l.match(/(licenciatura|ingenier[íi]a|medicina|arquitectura|abogac[íi]a|carrera|tecnicatura|profesorado|contadur[íi]a|psicolog[íi]a|dise[ñn]o|comunicaci[óo]n|inform[áa]tica|sistemas|biolog[íi]a|qu[íi]mica|f[íi]sica|matem[áa]tica|econom[íi]a|administraci[óo]n|turismo|periodismo|enfermer[íi]a|educaci[óo]n|filosof[íi]a|historia|geograf[íi]a|arte|m[úu]sica|teatro|danza|deporte|ciencias|computaci[óo]n|electr[óo]nica|civil|industrial|qu[íi]mica|mec[áa]nica|naval|aeron[áa]utica|petrolera|ambiental|gen[ée]tica|nutrici[óo]n|odontolog[íi]a|veterinaria|farmacia|bioqu[íi]mica|kinesiolog[íi]a|fisioterapia|logopedia|fonoaudiolog[íi]a|trabajo social|relaciones|recursos humanos|marketing|publicidad|gastronom[íi]a|hoteler[íi]a|turismo|bibliotecolog[íi]a|antropolog[íi]a|sociolog[íi]a|criminolog[íi]a|seguridad|defensa|militar|polic[íi]a|aduana|transporte|log[íi]stica|aviaci[óo]n|marina|pesca|agronom[íi]a|forestal|zootecnia|veterinaria|agropecuaria|enolog[íi]a|viticultura|horticultura|fruticultura|apicultura|ac[úu]icultura|pesquer[íi]a|minas|petroleo|gas|energ[íi]a|renovable|nuclear|espacial|astronom[íi]a|astrof[íi]sica|geolog[íi]a|meteorolog[íi]a|oceanograf[íi]a|hidrolog[íi]a|climatolog[íi]a|paleontolog[íi]a|arqueolog[íi]a|ling[üu][íi]stica|traducci[óo]n|interpretaci[óo]n|letras|literatura|filolog[íi]a|edici[óo]n|editorial|periodismo|comunicaci[óo]n|bibliotecolog[íi]a|documentaci[óo]n|archivolog[íi]a|museolog[íi]a|gesti[óo]n cultural|gesti[óo]n patrimonial|turismo|hotelera|gastronom[íi]a|enolog[íi]a|viticultura|fruticultura|horticultura|apicultura|ac[úu]icultura|pesquer[íi]a|minas|petroleo|gas|energ[íi]a|renovable|nuclear|espacial|astronom[íi]a|astrof[íi]sica|geolog[íi]a|meteorolog[íi]a|oceanograf[íi]a|hidrolog[íi]a|climatolog[íi]a|paleontolog[íi]a|arqueolog[íi]a|ling[üu][íi]stica|traducci[óo]n|interpretaci[óo]n|letras|literatura|filolog[íi]a|edici[óo]n|editorial|periodismo|comunicaci[óo]n|bibliotecolog[íi]a|documentaci[óo]n|archivolog[íi]a|museolog[íi]a|gesti[óo]n cultural|gesti[óo]n patrimonial|turismo|hotelera|gastronom[íi]a|enolog[íi]a|viticultura|fruticultura|horticultura|apicultura|ac[úu]icultura|pesquer[íi]a|minas|petroleo|gas|energ[íi]a|renovable|nuclear|espacial|astronom[íi]a|astrof[íi]sica|geolog[íi]a|meteorolog[íi]a|oceanograf[íi]a|hidrolog[íi]a|climatolog[íi]a|paleontolog[íi]a|arqueolog[íi]a|ling[üu][íi]stica|traducci[óo]n|interpretaci[óo]n|letras|literatura|filolog[íi]a|edici[óo]n|editorial|periodismo|comunicaci[óo]n|bibliotecolog[íi]a|documentaci[óo]n|archivolog[íi]a|museolog[íi]a|gesti[óo]n cultural|gesti[óo]n patrimonial)/i));
        return carreras.length > 0
            ? `Carreras en ${universidad}:\n- ${carreras.join("\n- ")}`
            : `No se encontraron carreras para ${universidad}.`;
    }
});

// Listar universidades que dictan una carrera (búsqueda textual simple)
const listarUniversidadesDeCarreraTool = tool({
    name: "listarUniversidadesDeCarrera",
    description: "Lista universidades que mencionan una carrera en su bloque de texto (búsqueda textual simple).",
    parameters: z.object({ carrera: z.string() }),
    execute: async ({ carrera }) => {
        if (!pdfTexto) await procesarPDF();
        const universidades = Array.from(universidadesSet);
        const universidadesConCarrera = universidades.filter(u => {
            const regex = new RegExp(`(^|\n)(.*${u}.*)(\n[\s\S]{0,1000})`, 'i');
            const match = pdfTexto.match(regex);
            if (!match) return false;
            const bloque = match[0];
            return bloque.toLowerCase().includes(carrera.toLowerCase());
        });
        return universidadesConCarrera.length > 0
            ? `Universidades que dictan ${carrera}:\n- ${universidadesConCarrera.join("\n- ")}`
            : `No se encontraron universidades que dicten ${carrera}.`;
    }
});

// --- Agente con todas las tools ---
const elAgente = agent({
    tools: [
        buscarPorUniversidadTool,
        buscarPorCarreraTool,
        listarUniversidadesTool,
        listarCarrerasTool,
        buscarUniversidadPorPalabraTool,
        detalleUniversidadTool,
        compararUniversidadesTool,
        buscarCarreraPorPalabraTool,
        detalleCarreraTool,
        compararCarrerasTool,
        listarCarrerasDeUniversidadTool,
        listarUniversidadesDeCarreraTool
    ],
    llm: ollamaLLM,
    systemPrompt: systemPrompt,
});



// Solo iniciar el chat si este archivo es el entrypoint principal
if (
    process.argv[1]?.toLowerCase().includes('main.js') ||
    import.meta.url?.toLowerCase().includes('main.js')
  ) {
    empezarChat(elAgente, mensajeBienvenida);
  }

export { elAgente };