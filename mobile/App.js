// mobile/App.js

import { registerRootComponent } from 'expo'; // 👈 第1步：导入“登记员”
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';

function App() { // 👈 第2步：把你的组件从默认导出变成一个普通函数
  return (
    <View style={styles.container}>
      <Text>我们的App从这里开始!</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

registerRootComponent(App); // 👈 第3步：调用“登记员”，把你的App组件登记为根组件