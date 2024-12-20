/**
 * React Native zeshaApp
 * https://github.com/facebook/react-native
 *
 * @format
 * @flow strict-local
 */

import React from 'react';

import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  Dimensions,
  TouchableOpacity,
  ScrollView,
  StatusBar
} from 'react-native';

import{
  RTCPeerConnection,
  RTCIceCandidate,
  RTCSessionDescription,
  RTCView,
  MediaStream,
  MediaStreamTrack,
  mediaDevices,
  registerGlobals
} from 'react-native-webrtc'

import io from 'socket.io-client'

const dimensionx = Dimensions.get('window').width
const dimensiony = Dimensions.get('window').height

class App extends React.Component {

  constructor(props){
    super(props)

    this.state = {
      localStream: null,
      remoteStream: null,
    }

    this.pc = null
    this.sdp = null
    this.socket = null
    this.candidates = []
  }

  componentDidMount = () => {

    this.socket = io.connect(
      'https://adb3-70-137-105-159.ngrok-free.app/webRTCPeers',  //https://aafb-70-137-105-159.ngrok-free.app
      {
        path: '/webrtc',
        query: {}
      }
    )
    // connection is successful
    this.socket.on('connection-success', success => {
      //console.log('***URL: ', this.socket.toURL())
      console.log(success)
    })

    // sdp received
    this.socket.on('sdp', data => {
      this.sdp = JSON.stringify(data.sdp)
      console.log('****', this.sdp.type);
      if(data.sdp.type === 'offer' || data.sdp.type === 'answer'){   
        this.pc.setRemoteDescription(new RTCSessionDescription(data.sdp))
        console.log("Incomming Call...")
      }else{
        console.log("Call Established.")
      }
    })

    // candidate is received
    this.socket.on('candidate', (candidate) =>{
      this.candidates = [...this.candidates, candidate]
      this.pc.addIceCandidate(new RTCIceCandidate(candidate))
    })

    const pc_config = {
      'iceServers' : [
        {
          urls: 'stun:stun.l.google.com:19302'
        }
      ]
    }

    this.pc = new RTCPeerConnection()
    this.pc.setConfiguration(pc_config)

    this.pc.getStats().then(desc => console.log('PeerConnection stats:',desc))

    this.pc.onicecandidate = (e) => {
      if(e.candidate) {
        console.log(JSON.stringify(e.candidate))
        this.sendToPeer('candidate', e.candidate)
      }
    }

    this.pc.oniceconnectionstatechange = (e) => {
      console.log(e)
    }

    this.pc.ontrack = (e) => {
      // got remote stream
      debugger

      console.log('ontrack method call')
      this.setState({
        remoteStream : e.streams[0]
      })
    }

    // successfully received stream
    const works = (stream) =>{
      console.log('Inside success method of getUserMedia function with streamURL:', stream.toURL())
      this.setState({
        localStream : stream
      })
      stream.getTracks().forEach(track => this.pc.addTrack(track, stream))
    }

    const fails = (e) => {
      console.log('Failed getUserMedia function with Error: ', e)
    }

    let isFront = true;
    mediaDevices.enumerateDevices().then(sourceInfos => {
      console.log('SourceInfos:', sourceInfos);
      let videoSourceId;
      for (let i =0 ; i < sourceInfos.letgth; i++) {
        const sourceInfo = sourceInfos[i];
        if(sourceInfo.kind == "videoinput" && sourceInfo.facing == (isFront ? "front" : "environment")) {
          videoSourceId = sourceInfo.deviceId
        }
      }

      const constraints = {
        audio: true,
        video: {
          mandatory: {
            minWidth: 500,   // custom width, height and framerate
            minHeight: 300,
            minFrameRate: 30
        },
        facingMode: (isFront ? "user" : "environment"),
        optional: (videoSourceId ? [{ sourceId: videoSourceId}] : [])
      }
    }
      // const constraints_ ={audio: true, video:true}
      console.debug(constraints)

      mediaDevices.getUserMedia(constraints)
      .then(works)
      .catch(fails);
    });
  }

  // send to peer
   sendToPeer = (eventType, payload) => {
    this.socket.emit(eventType, payload)
  }

   processSDP = (sdp) => {
     console.log(JSON.stringify(sdp))
     this.pc.setLocalDescription(sdp)
     this.sendToPeer( 'sdp', { sdp })
    }

  createOffer = () => {
    console.log('Inside create offer function.')

    this.pc.createOffer({
      offerToReceiveAudio: 1,
      offerToReceiveVideo: 1,
    }).then(sdp =>{
      this.processSDP(sdp)
    }).catch(e=>console.debug('Error creating offer', e))
  }

  // create answer
  createAnswer = () => {
    console.log('Inside create answer function.')

    this.pc.createAnswer({ 
      offerToReceiveAudio: 1,
      offerToReceiveVideo: 1,
    }).then(sdp => {
      this.processSDP(sdp)
    }).catch(e => console.debug('Error creating answer', e))
  }

  addCandidate = () => {
    this.candidates.forEach(candidate => {
      console.log(JSON.stringify(candidate))
      this.pc.addIceCandidate(new RTCIceCandidate(candidate))
    });
  }
  
 render(){

  const {
    localStream,
    remoteStream
  } = this.state

  const remoteVideo = remoteStream ?
  (
    <RTCView
      key = {2}
      mirror = {true}
      style = {{...styles.rtcViewRemote}}
      objectFit='cover'
      streamURL={remoteStream && remoteStream.toURL()}
    />
  ) :
  (
    <View style={{ padding : 15,}}>
      <Text style={{fontSize:22, textAlign: 'center', color: 'white'}}>waiting...</Text>
    </View>
  )

  return (
    <SafeAreaView style={{flex: 1,}}>
      <StatusBar backgroundColor='blue' barStyle={'dark-content'}/>
      <View style={{...styles.buttonsContainer}}>
        <View style={{flex: 1, }}>
          <TouchableOpacity onPress={this.createOffer}> 
          <View style={styles.button}>
            <Text style={{...styles.textContent, }}>Call</Text>
          </View >
          </TouchableOpacity>
        </View>
        <View style={{flex: 1,}}>
          <TouchableOpacity onPress={this.createAnswer}> 
          <View style={styles.button}>
            <Text style={{...styles.textContent, }}>Answer</Text>
          </View>
          </TouchableOpacity>
        </View>
      </View>
      <View style={{...styles.videosContainer, }}>
      <View style= {{
        position: 'absolute',
        zIndex: 1000,
        bottom: 10,
        right: 10,
        width: 100, 
        height: 200,
        backgroundColor: 'black',
      }}>
        <View style={{flex: 1,}}>
          <TouchableOpacity onPress={() => localStream._tracks[1]._switchCamera()}>
            <View>
              <RTCView 
              key={1}
              zOrder={0}
              objectFit='cover'
              style={{...styles.rtcView}}
              streamURL={localStream && localStream.toURL()}
              /> 
            </View>
          </TouchableOpacity>
        </View>
      </View>  
      <ScrollView style={{ ...styles.scrollView}}>
        <View style={{
          flex: 1,
          width:'100%',
          backgroundColor: 'black',
          justifyContent:'center',
          alignItems: 'center',
        }}>
       { remoteVideo }
       </View>
      </ScrollView>
      </View>
    </SafeAreaView>
  );
 }
};

const styles = StyleSheet.create({
  buttonsContainer: {
    flexDirection: 'row',
  },
  button: {
    margin: 5,
    paddingVertical: 10,
    backgroundColor: 'lightgray',
    borderRadius: 5,
  },
  textContent: {
    fontFamily: 'Avenir',
    fontSize: 20,
    textAlign: 'center',
  },
  videosContainer: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  rtcView: {
    width: 100,
    height: 200,
    backgroundColor: 'black',
  },
  scrollView: {
    flex: 1,
    backgroundColor: 'teal',
    padding: 15,
  },
  rtcViewRemote: {
    width: dimensionx - 10,
    backgroundColor: 'black',
    height: dimensiony - 10,
  }
});

export default App;
