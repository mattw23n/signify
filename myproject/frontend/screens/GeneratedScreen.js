import React, { useRef, useState, useEffect } from 'react';
import { View, Text, StatusBar, ScrollView, TouchableOpacity, TextInput } from 'react-native';
import { Video } from 'expo-av';
import * as Clipboard from 'expo-clipboard';

const GeneratedScreen = ({ route }) => {
    const { videoUri } = route.params || {};
    const videoRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(true);

    const handlePlayPause = async () => {
        if (isPlaying) {
        await videoRef.current.pauseAsync();
        setIsPlaying(false);
        } else {
        await videoRef.current.playAsync();
        setIsPlaying(true);
        }
    };

    const [copied, setCopied] = React.useState('');
    const [height, setHeight] = useState(40);

    const handleContentSizeChange = (event) => {
        setHeight(event.nativeEvent.contentSize.height);
    };

    let genText =
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum';
    let [text, setText] = useState(genText);

    /*
    TODO
    add text fetching from API here
    */

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
    <View className='flex-1 bg-white pt-6 pb-36'>
        <View style={{ flexDirection: 'row', alignItems: 'center'}}>
            <Text className='pl-9 pb-4
                            text-lg font-black mb-1.2'>
                            Generated Text
                            </Text>
            <TouchableOpacity className='pl-24 pb-4' onPress={copyToClipboard}>
                {copied?
                (<Text style={{ fontSize: 16, color: '#32E08C' }}>
                    Copied!
                </Text>) :
                (<Text style={{ fontSize: 16, color: '#A5A5A5' }}>
                    Copy
                </Text>
                )}
            </TouchableOpacity>
        </View>

        <StatusBar style='auto'/>
            <View className='rounded-lg bg-gray-100
                            w-72 pb-2
                            mx-auto'>
                <ScrollView vertical={true}
                            maxHeight={200}>
                    <View className='pl-4'>
                        <TextInput
                            value={text}
                            onChangeText={(newText) => setText(newText)}
                            style={{ width: 260, textAlignVertical: 'top',
                                    paddingVertical: 8, fontSize: 15, height}}
                            multiline={true}
                            onContentSizeChange={handleContentSizeChange}
                        />
                    </View>
                </ScrollView>
            </View>

        <View className='pt-4 pb-4 items-center'>
            <Video
            ref={videoRef}
            source={{ uri: videoUri }}
            rate={1.0}
            volume={1.0}
            isMuted={false}
            resizeMode='contain'
            shouldPlay={isPlaying}
            isLooping
            style={{ width: '75%', height: '75%'}}
        />
            <TouchableOpacity
                style={{
                    backgroundColor: '#32E08C',
                    paddingVertical: 8,
                    paddingHorizontal: 28,
                    borderRadius: 24,
                    marginTop: 12
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
    </View>
  );
};

export default GeneratedScreen;