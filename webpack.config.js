const path = require('path');
//var nodeExternals = require('webpack-node-externals');

module.exports = {
  context: path.resolve(__dirname, 'src'),
  entry: {
    index: ['./index.js'],
  },
  output: {
    path: path.join(__dirname, '/dist'),
    filename: 'bundle.js',
    libraryTarget: "umd"
  },
  module: {
    rules: [
      //{
      //  test: /\.worker\.js$/,
      //  use: {
      //    loader: 'worker-loader',
      //    options: { inline: true }
      //  },
      //},
      {
        test: /\.js$/,
        exclude: /(node_modules)/,
        use: {
          loader: 'babel-loader'
        },
      },
    ]
  },
//  target: 'node',
//  externals: [nodeExternals({whitelist: [/^long/, /^protobufjs/, /^@protobufjs/, /^seedrandom/, /^@tensorflow/, /^@types/]})],
};
