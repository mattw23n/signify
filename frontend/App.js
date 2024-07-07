import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { SafeAreaView, Image } from "react-native";
import NaviBar from "./components/NaviBar";
import GeneratedScreen from "./screens/GeneratedScreen";
import Header from "./components/HowItWorks";
import { View } from "react-native";

const Stack = createNativeStackNavigator();

const App = () => {
  return (
      <NavigationContainer>
        <Stack.Navigator
          screenOptions={{
            headerTitleAlign: 'center',
            headerTitle: () => (
              <View style={{ width: 120, height: 40, justifyContent: 'center', alignItems: 'center' }}>
                <Image 
                  source={require('./assets/icon.png')} 
                  style={{ width: '100%', height: '100%' }} 
                  resizeMode="contain" 
                />
              </View>
            ),
            headerRight: () => (
              <View style={{ flexDirection: 'row', alignItems: 'center', marginRight: 10 }}>
                <Header />
              </View>
            ),
            headerStyle: {
              height: 50,
            },
            headerBackground: () => (
              <View>
                <Image 
                  source={require('./assets/top.png')} 
                  style={{ width: '100%', height: '100%' }} 
                  resizeMode="cover" 
                />
              </View>
              
            ),
          }}>
        <Stack.Screen
          name="NaviBar"
          component={NaviBar}
        />
        <Stack.Screen
            name="Generated"
            component={GeneratedScreen}
        />
        </Stack.Navigator>
      </NavigationContainer>
  );
};

export default App;