import express from "express";
import cors from "cors";
import { elAgente } from "./main.js"; // Asegurate de exportar elAgente desde main.js

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

app.post("/api/chat", async (req, res) => {
  const { mensaje } = req.body;
  if (!mensaje) {
    return res.status(400).json({ error: "Mensaje requerido" });
  }
  try {
    const respuesta = await elAgente.run(mensaje);

    let texto = respuesta.data?.result || respuesta.result || "";
    if (texto.includes("</think>")) {
      texto = texto.split("</think>").pop().trim();
    }

    res.json({ respuesta: texto });
    
  } catch (error) {
    console.error("Error en /api/chat:", error); // <-- Esto es clave
    res.status(500).json({ error: "Error procesando el mensaje" });
  }
});

app.listen(PORT, () => {
  console.log(`Servidor escuchando en http://localhost:${PORT}`);
});