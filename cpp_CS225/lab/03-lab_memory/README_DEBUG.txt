BUG 1:

1. Using gdb, find the program fail at the 
	statement_a: Allocator theAllocator("students.txt", "rooms.txt");
2. Using gdb, find the statement_a fails at 
	statement_a_1: loadRooms(roomFile);
3. Using gdb, find the statement_a_1 fails at
	rooms[i] = fileio::nextRoom();

Since there is no error in fileio, so it must be wrong with the rooms inintialization.
    1. roomCount is not initialized, so the 
        new Room[roomCount];
       won't work
    2. Index out of boundary
        i++

BUG 2:

After finishing BUG 1, we can run the exe by 
    ./allocate

However, if you use
    valgrind --leak-check=full ./allocate

The first error you'll get is:

    ==16050== Invalid write of size 8
    ==16050==    at 0x403539: Room::addLetter(Letter const&) (room.
    cpp:46)
    ==16050==    by 0x404EB6: Allocator::solve() (allocator.cpp:94)
    ==16050==    by 0x404964: Allocator::allocate() (allocator.cpp:
    70)
    ==16050==    by 0x422A84: main (main.cpp:26)

it can be out of index or not initialize the object properly. 
Here, the issue is in the copy constructor of "Room", 
where the copy process of "letters" is not correct.

After fixing this issue, you check again. The very next bug you will get the following

    ==17018==
    ==17018== HEAP SUMMARY:
    ==17018==     in use at exit: 2,592 bytes in 11 blocks
    ==17018==   total heap usage: 658 allocs, 647 frees, 57,680 bytes allocated
    ==17018==
    ==17018== 208 bytes in 1 blocks are definitely lost in loss record 1 of 3
    ==17018==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
    ==17018==    by 0x4EC38C9: operator new(unsigned long) (in /usr/lib/x86_64-linux-gnu/libc++.so.1.0)
    ==17018==    by 0x4044C2: Allocator::createLetterGroups() (allocator.cpp:24)
    ==17018==    by 0x404473: Allocator::Allocator(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&) (allocator.cpp:16)
    ==17018==    by 0x422B4B: main (main.cpp:25)


This is as the result of not free the heap memory claimed by the private memebr "alpha" of the allocator. So you need define your own destructor.

After fixing the issue, you check again. Then the next bug is

    ==17368==
    ==17368== HEAP SUMMARY:
    ==17368==     in use at exit: 2,384 bytes in 10 blocks
    ==17368==   total heap usage: 658 allocs, 648 frees, 57,
    680 bytes allocated
    ==17368==
    ==17368== 2,384 (512 direct, 1,872 indirect) bytes in 1
    blocks are definitely lost in loss record 2 of 2
    ==17368==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/
    vgpreload_memcheck-amd64-linux.so)
    ==17368==    by 0x4EC38C9: operator new(unsigned long) (
    in /usr/lib/x86_64-linux-gnu/libc++.so.1.0)
    ==17368==    by 0x4047AA: Allocator::loadRooms(std::__1:
    :basic_string<char, std::__1::char_traits<char>, std::__
    1::allocator<char> > const&) (allocator.cpp:48)
    ==17368==    by 0x40448D: Allocator::Allocator(std::__1:
    :basic_string<char, std::__1::char_traits<char>, std::__
    1::allocator<char> > const&, std::__1::basic_string<char
    , std::__1::char_traits<char>, std::__1::allocator<char>
     > const&) (allocator.cpp:18)
    ==17368==    by 0x422B8B: main (main.cpp:25)
  
This error is the same type as mentioned above.