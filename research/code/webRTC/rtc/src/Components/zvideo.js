import React, { Component} from "react";

class Video extends Component {
    constructor(props){
        super(props)
        this.state = {}
    }

    compomponentWillReceiveProps(nextProps) {
        this.video.srcObject = nextProps.videoStream
    }

    render(){
        return(
            <div style={{...this.props.fromStyle}} >
                <video
                    id={this.props.id}
                    muted={this.props.muted}
                    autoPlay
                    style={{...this.props.videoStyle}}
                    //ref={this.props.videoRef}
                    ref= { (ref) => {this.video = ref}}
                />  
            </div>
        )
    }
}

export default Video