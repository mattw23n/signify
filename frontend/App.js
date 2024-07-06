import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { SafeAreaView, Image } from "react-native";
import NaviBar from "./components/NaviBar";
import GeneratedScreen from "./screens/GeneratedScreen";

const Stack = createNativeStackNavigator();

const App = () => {
  return (
      <NavigationContainer>
        <Stack.Navigator
          screenOptions={{
            headerTitle: "",
            headerStyle: {
              height: 50,
            },
            headerBackground: () => (
              <SafeAreaView className='flex-1'>
                <Image
                source={require("./assets/topBar.png")}
                style={{ height: "100%", width: "100%"}}
               />
              </SafeAreaView>
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