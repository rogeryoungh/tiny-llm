add_rules("plugin.compile_commands.autoupdate")
add_rules("mode.release", "mode.debug")

target("tiny-llm")
	set_kind("binary")
	set_languages("c++23")
	add_files("src/**.cpp")
