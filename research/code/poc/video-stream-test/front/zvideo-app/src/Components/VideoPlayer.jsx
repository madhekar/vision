import React, {useRef, useEffect} from 'react'

const VideoPlayer = ({videoId}) => {
    const videoRef = useRef(null)

    useEffect(() => {
        if (videoRef.current){
            videoRef.current.pause()
            videoRef.current.removeAttribute('src')
            videoRef.current.load()
        }
    })
  return (
    <video ref={videoRef} width='720'  controls autoplay> 
       <source src={`http://localhost:3000/videos/${videoId}`} type='video/mp4'></source>
       Your browser does not support tag video.
    </video>
  )
}

export default VideoPlayer