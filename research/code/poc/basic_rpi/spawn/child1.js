const spawn = require('child_process').spawn;
const stream = require('stream')
const events_1 = require("events")

function runIt(cmd, args) {
 return new Promise(
    function(resolve, reject) {
        const child_ = spawn(cmd, args);

        // Edit thomas.g: My child process generates binary data so I use buffers instead, see my comments inside the code 
        // Edit thomas.g: let stdoutData = new Buffer(0)
        //let stdoutData = "";
        let stdoutData = new Buffer.alloc(0)
        let stderrData= "";

        child_.stdout.on('data', (data) => {
        // Edit thomas.g: stdoutData = Buffer.concat([stdoutData, chunk]);
            //stdoutData += data;
            stdoutData = Buffer.concat([stdoutData, data])
        });

        child_.stderr.on('data', (data) => {
            stderrData += data;
        });

        child_.on('close', (code) => {
            if (stderrData){
                reject(stderrData);
            } else {
                resolve(stdoutData);
            }
        });
        child_.on('error', (err) => {
            reject(err);
        });
    }) 
}
/**** 
usage
****/
// runIt('libcamera-vid', ['--framerate', '30', '--timeout', '0', '--nopreview', '--output', '-'])
runIt('ls', ['-al', '..'])
.then(function(stdoutData) {
   console.log(`data: ${stdoutData}`)
}, function(err) {
    console.log(`error: ${err}`)
});