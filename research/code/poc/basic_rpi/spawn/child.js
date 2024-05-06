var cp = require('child_process');

var progs = {
    lists: 'ls',
    copy: 'cp',
    folder: 'mkdir'
}

var child = cp.spawn(progs.lists, ['-r', '/Library']);

child.stdout.on('data', (data) => {
   console.log(`data:\n ${data}`)
})