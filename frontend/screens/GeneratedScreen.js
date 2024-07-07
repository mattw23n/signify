import React, { useRef, useState, useEffect } from 'react';
import { View, Text, StatusBar, ScrollView, TouchableOpacity, TextInput, Image } from 'react-native';
import { Video, Audio } from 'expo-av';
import * as Clipboard from 'expo-clipboard';
import axios from 'axios';

const GeneratedScreen = ({ route }) => {
    const { videoUri } = route.params || {};
    const videoRef = useRef(null);

    const [isPlaying, setIsPlaying] = useState(true);
    const [isAudioPlaying, setAudioPlaying] = useState(false);
    const [sound, setSound] = useState(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [copied, setCopied] = React.useState('');
    const [height, setHeight] = useState(80);
    const [text, setText] = useState(null);

    useEffect(() => {
        return () => {
        if (sound) {
            sound.unloadAsync();
        }
        };
    }, [sound]);

    const loadAndPlayAudio = async () => {
        try {
        const { sound: newSound } = await Audio.Sound.createAsync(
            { uri: 'INSERT_YOUR_IP_ADDRESS_HERE:8000/api/audio/' },
            { shouldPlay: true }
        );
        setSound(newSound);
        setAudioPlaying(true);
        setIsLoaded(true);

        newSound.setOnPlaybackStatusUpdate((status) => {
            if (status.didJustFinish) {
            setAudioPlaying(false);
            }
        });
        } catch (error) {
        console.error('Error fetching or playing audio file:', error);
        }
    };

    const playPauseAudio = async () => {
        if (!isLoaded) {
        await loadAndPlayAudio();
        } else {
        if (isAudioPlaying) {
            await sound.pauseAsync();
            setAudioPlaying(false);
        } else {
            await sound.playAsync();
            setAudioPlaying(true);
        }
        }
    };

    const handlePlayPause = async () => {
        if (isPlaying) {
        await videoRef.current.pauseAsync();
        setIsPlaying(false);
        } else {
        await videoRef.current.playAsync();
        setIsPlaying(true);
        }
    };

    const handleContentSizeChange = (event) => {
        setHeight(event.nativeEvent.contentSize.height + 20);
    };

    useEffect(() => {
        axios.get('INSERT_YOUR_IP_ADDRESS_HERE/api/caption/')
          .then(response => {
            console.log('Response data:', response.data);
            setText(response.data.results);
          })
          .catch(error => {
            console.error('Error fetching results:', error);
          });
      }, []);

    const copyToClipboard = async () => {
        await Clipboard.setStringAsync(text);
        setCopied(true);
    };

    useEffect(() => {
        setCopied(false); 
      }, [text]);

    if (!videoUri) {
        return <Text className='pt-72 text-center
                text-lg font-bold mb-1.2
                text-gray-400'>
                No video URI provided
        </Text>;
    }

    return (
    <View className='flex-1 bg-white'>
        <View className='items-center'>
            <Video
                ref={videoRef}
                source={{ uri: videoUri }}
                rate={1.0}
                volume={0.0}
                isMuted={false}
                resizeMode='contain'
                shouldPlay={isPlaying}
                isLooping
                style={{ width: 330, height: 200}}
                className = "rounded-xl mx-10 my-5"
                on
            />
            <TouchableOpacity
                className = "py-2 px-10 rounded-xl mb-8"
                style={{
                    backgroundColor: '#32E08C',
                  }}
                onPress={handlePlayPause}>
                {isPlaying ?
                (<Text className='font-black text-base text-white'>
                    Pause
                </Text>) :
                (<Text className='font-black text-base text-white'>
                    Play
                </Text>)}
             </TouchableOpacity>
        </View>

        <View className="flex flex-row items-center justify-between mr-10 mb-5">
            <Text className='pl-10 pb-4
                            text-xl font-black'>
                            Text to Speech
            </Text>
            <TouchableOpacity
                className = "py-2 px-2 rounded-xl mb-3"
                style={{
                        backgroundColor: '#32E08C',
                    }}
                    onPress={playPauseAudio}>
                    <Image
                    className="w-8 h-8"
                    source={
                        isAudioPlaying
                        ? require('../assets/pause2.png')
                        : require('../assets/play2.png')
                    }
                    />
            </TouchableOpacity>
        </View>
    
        <View style={{ flexDirection: 'row', alignItems: 'center'}}>
            <Text className='pl-10 pb-4 -pt-20
                            text-xl font-black mb-1.2'>
                            Generated Captions
                            </Text>
        </View>

        <StatusBar style='auto'/>
            <View className='rounded-xl bg-gray-200
                            w-90 pb-2
                            mx-10'>
                <ScrollView vertical={true}
                            maxHeight={200}>
                    <View className='pl-4'>
                        <TextInput
                            value={text}
                            onChangeText={(newText) => setText(newText)}
                            style={{ width: 260, textAlignVertical: 'top',
                                    paddingVertical: 8, fontSize: 15, height: height}}
                            multiline={true}
                            onContentSizeChange={handleContentSizeChange}
                        />
                    </View>
                </ScrollView>
            </View>

        <TouchableOpacity
                className = "mx-20 my-10 justify-center border border-gray-500 p-4 rounded-xl"
                onPress={copyToClipboard}>
                {copied?
                (<Text style={{ fontSize: 16, color: '#32E08C' }} className="text-center font-bold">
                    Copied!
                </Text>) :
                (<Text style={{ fontSize: 16}} className="text-center font-bold">
                    Copy
                </Text>
                )}
        </TouchableOpacity>
    </View>
  );
};

export default GeneratedScreen;