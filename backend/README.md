# Entregable: Sistema de Gestión de Estudiantes con LLM Tools

Este proyecto es un entregable para implementar un sistema de gestión de estudiantes utilizando LLMs (Large Language Models) y Tools.

## 📚 Requisitos Previos

- Node.js >= 18
- [Ollama](https://ollama.com/) instalado y corriendo
- Modelo qwen3:1.7b instalado

## 🛠 Instalación

1. Instala las dependencias:
```bash
npm install
```

2. Asegúrate de tener Ollama corriendo y el modelo instalado:
```bash
ollama run qwen3:1.7b
```

## 🎯 Tu Tarea

Debes implementar las siguientes funcionalidades:

1. En `src/lib/estudiantes.js`:
   - Método `agregarEstudiante(nombre, apellido, curso)`
   - Método `buscarEstudiantePorNombre(nombre)`
   - Método `buscarEstudiantePorApellido(apellido)`
   - Método `listarEstudiantes()`

2. En `src/ejemplo-alumnos-tools.js`:
   - Tool `buscarPorNombre`
   - Tool `buscarPorApellido`
   - Tool `agregarEstudiante`
   - Tool `listarEstudiantes`

## 💡 Ayuda

- Cada método y Tool tiene comentarios TODO indicando dónde implementar el código
- Revisa la documentación de llamaindex para entender cómo funcionan las Tools
- Utiliza la clase `Estudiantes` para manejar los datos
- Las Tools deben usar los métodos de la clase `Estudiantes`

## 🚀 Para Ejecutar

```bash
npm start
```

## 📝 Notas

- El código base ya incluye:
  - Interfaz CLI funcional
  - Formateo de respuestas
  - Manejo básico de archivos
  - Estructura del proyecto

- No modifiques:
  - La estructura de los archivos
  - Los nombres de los métodos/Tools
  - Los parámetros definidos

## 📚 Recursos

- [Documentación de LlamaIndex](https://docs.llamaindex.ai/)
- [Documentación de Zod](https://zod.dev/)
- [Ejemplos de Tools](https://docs.llamaindex.ai/en/stable/examples/tools/)
