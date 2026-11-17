#!/usr/bin/ruby
# Serve this folder over HTTPS on your Wi-Fi so a phone can use the microphone.
# Usage: /usr/bin/ruby ./start-phone-server.rb 8443
# Uses a throw-away self-signed certificate: the phone will show a warning you must accept.
# Needs the macOS system Ruby (which bundles WEBrick); on newer Rubies run `gem install webrick`.
require 'webrick'
require 'webrick/https'
require 'socket'

port = (ARGV[0] || 8443).to_i
ip = Socket.ip_address_list.find { |a| a.ipv4_private? }
host = ip ? ip.ip_address : 'localhost'
server = WEBrick::HTTPServer.new(
  Port: port,
  DocumentRoot: File.dirname(File.expand_path(__FILE__)),
  SSLEnable: true,
  SSLCertName: [['CN', host]]
)
server.mount_proc('/__ping') { |_req, res| res.body = 'ok' }
WEBrick::HTTPUtils::DefaultMimeTypes.store('js', 'text/javascript')
WEBrick::HTTPUtils::DefaultMimeTypes.store('json', 'application/json')
trap('INT') { server.shutdown }
puts "\nOpen on your phone (same Wi-Fi):  https://#{host}:#{port}/\n\n"
server.start
