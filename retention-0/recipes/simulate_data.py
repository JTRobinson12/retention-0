import keys
import piping
import simulate

people = simulate.people()
projects = simulate.projects()

piping.raw_data.write(people, keys.Imported.PEOPLE)
piping.raw_data.write(projects, keys.Imported.PROJECTS)
piping.raw_data.write(simulate.assignments(people, projects), keys.Imported.ASSIGNMENTS)
