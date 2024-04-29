const express = require('express')
const fs = require('fs')

const app = express()
const port = 3000
const vFileMap = {
    'd1' : 'videos/dance1.mp4',
    'd2' : 'videos/dance2.mov',
    'd3' : 'videos/dance3.mov',
    'd4' : 'videos/dance4.mov'
   }
//video service...
app.get('/videos/:filename', (req, res) => {
     const filename = req.params.filename
     const filepath = vFileMap[filename]

     if (!filepath){
        return res.status(404).send('File not found')
     }

     const stat = fs.statSync(filepath);
     const fileSize = stat.size;
     const range = req.headers.range;

     if (range){
        const parts = range.replace(/bytes=/, '').split('-')
        const start = parseInt(parts[0], 10)
        const end = parts[1] ? parseInt(parts[1], 10) : fileSize -1 ;

        const chunksize =  end - start + 1
        const file = fs.createReadStream(filepath, {start, end})
        const head = {
            'Content-Range' : `bytes ${start} - ${end}/${fileSize}`,
            'Accept-Ranges' : 'bytes',
            'Content-Length' : chunksize,
            'Content-Type' : 'video/mp4'
        };
        res.writeHead(206, head);
        file.pipe(res);
      }
      else
      {
        const head = {
            'Content-Length' : fileSize,
            'Content-Type' : 'video/mp4'
        };
        res.writeHead(200, head);
        fs.createReadStream(filepath).pipe(res);
     }
})

app.listen(port, () => {
    console.log(`video service is rinnging at port: ${port}`)
})