var fs = require('fs')
var request = require('request')

queue = []
for (i=0; i<=500; i++){
queue.push([`https://picsum.photos/id/${i+1}/100/100`, i + 1])
}

start = Date.now()
promises = []

queue.forEach(url => {
    promises.push(
       new Promise(resolve => {
           request(url[0]).pipe(fs.createWriteStream(`images/${url[1]}.png`)).on('close', () => {
                 console.log(`got picture ${url[1]} through a single thread via promise`)
	         resolve()
	  })
      }))
})

Promise.all(promises).then(() => {
	end = Date.now() - start
        colsole.log(`\n time taken: ${end / 1000}`)
})
