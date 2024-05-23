const Bridge = require("@molfar/py-bridge")
const path = require("path")

const WordEmbedding = class extends Bridge {
	constructor(config){
		super(config)
		this.use("__run", config.pythonScript)
	}

	getSentiments(message) {
		return this.__run(message)
	}
}

module.exports = WordEmbedding

// const run = async () => {

// 	const config = {
// 		mode: 'text',
// 		encoding: 'utf8',
// 		pythonOptions: ['-u'],
// 		pythonPath: (process.env.NODE_ENV && process.env.NODE_ENV == "production") ? 'python' : 'python.exe',
// 		args: path.resolve(__dirname,"../../model.ftz")
// 	}

// 	const analyser = new SentimentAnalyzer(config)
// 	analyser.start()

// 	const text = "Канал доступний за посиланням:\nhttps://t.me/UkraineNow".replace(/\n/," ")
// 	console.log(text)
// 	let res = await analyser.getSentiments({text})

// 	console.log("RESULT", res)
// }

// run()
