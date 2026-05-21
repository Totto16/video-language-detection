import type { Configuration } from 'webpack';


import { plugins } from './webpack.plugins';
import { AngularWebpackPlugin } from '@ngtools/webpack';

export const rendererConfig: Configuration = {
  module: {
    rules: [
      {
        test: /renderer\/preload.ts$/,
        use: {
          loader: 'ts-loader',
          options: {
            transpileOnly: true,
          },
        },
      },
      {
        test: /renderer\/.*\.[jt]s$/,
        loader: '@ngtools/webpack'
      },
      {
        test: /\.s[ac]ss$/i,
        use: [
          // Creates `style` nodes from JS strings
          { loader: 'style-loader' },
          // Translates CSS into CommonJS
          {
            loader: "css-loader"
          },
          // Compiles Sass to CSS
          { loader: "sass-loader" },
        ],
      }
    ],
  },
  plugins: [
    ...plugins,
    new AngularWebpackPlugin({
      tsconfig: './tsconfig.app.json'
    })
  ],
  resolve: {
    extensions: ['.js', '.ts', '.jsx', '.tsx', ".scss"],
  },
  target: "electron-renderer",

};
