import logo from './logo.svg';
import './App.css';
import {useRef} from 'react';

function App() {
  const localVideoRef = useRef()

  const getUserMedia = () => {
    const constraints = {
      audio : false,
      video : true,
    }
    navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => {
      // display stream
      localVideoRef.current.srcObject = stream
    })
    .catch(e => {
      console.log('getUserMedia error ...', e)
    })
  }

  return (
    <div style={{margin:10}}>
      <button onClick={() => getUserMedia()}>Get Media Access</button>
      <video ref={localVideoRef} autoPlay></video>
    </div>
  );
}

export default App;
