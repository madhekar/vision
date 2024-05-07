var cp = require('child_process');

var progs = {
    lists: 'ls',
    copy: 'cp',
    folder: 'mkdir',
    libcamera: 'libcamera-vid'
}

var child = cp.spawn(progs.lists, ['-l'], {cwd: '..'} );

var child = cp.spawn(progs.cp, ['test.txt', 'test1.txt'])

var child = cp.spawn(progs.folder, ['dummy'])

//var child = cp.spawn(progs.libcamera, ['--timeout', '0', '--framerate', '30', '--nopreview', '--output', '-']);
child.stdout.on('data', (data) => {
   console.log(`data:\n ${data}`)
})

child.stderr.on('data', (err) => {
    console.log(`error: ${err}`)
})

child.on('exit', () => {
    console.log('child process is finished!')
})

child.on('close', () =>{
    console.log('child process is closed!')
})