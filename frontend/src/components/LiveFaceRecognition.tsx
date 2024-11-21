import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import { io, Socket } from 'socket.io-client';

// Define types for recognized faces
interface RecognizedFace {
    name: string;
    location: number[]; // [top, right, bottom, left]
}

const FaceRecognitionComponent: React.FC = () => {
    const webcamRef = useRef<Webcam>(null);
    const [socket, setSocket] = useState<Socket | null>(null);
    const [recognizedFaces, setRecognizedFaces] = useState<RecognizedFace[]>([]);

    useEffect(() => {
        // Create a WebSocket connection to the server
        const newSocket = io('http://localhost:8000'); // Replace with your server URL
        setSocket(newSocket);

        // Listen for 'recognized_faces' event from server and update state
        newSocket.on('recognized_faces', (data: { faces: RecognizedFace[] }) => {
            setRecognizedFaces(data.faces); // Update the state with the recognized faces
        });

        return () => {
            newSocket.disconnect(); // Cleanup the socket connection when the component unmounts
        };
    }, []);

    useEffect(() => {
        // Automatically send frames every 1 second
        const intervalId = setInterval(() => {
            captureAndSendFrame();
        }, 1000);

        return () => {
            clearInterval(intervalId); // Cleanup interval on unmount
        };
    }, [socket]);

    const captureAndSendFrame = () => {
        // Ensure the webcam is ready and a screenshot can be taken
        const imageSrc = webcamRef.current?.getScreenshot();
        if (imageSrc && socket) {
            socket.emit('video_frame', { frame: imageSrc.split('base64,')[1] }); // Emit the frame to the server
        }
    };

    return (
        <div>
            <h1>Face Recognition</h1>

            {/* Webcam component */}
            <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                width="100%"
                videoConstraints={{
                    facingMode: 'user',
                }}
            />

            {/* Display recognized faces */}
            {recognizedFaces.length > 0 && (
                <div>
                    <h2>Recognized Faces:</h2>
                    <ul>
                        {recognizedFaces.map((face, index) => (
                            <li key={index}>
                                <strong>Name:</strong> {face.name} <br />
                                <strong>Location:</strong> {JSON.stringify(face.location)}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default FaceRecognitionComponent;
